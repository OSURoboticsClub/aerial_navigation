/*
 * Creator: Matthew Lei
 * Last updated: march 20 2015
 * Description:
 * main file for reading an image/video or capturing video feed and applying
 * various transformation filters to achieve wanted effect. Controlling what
 * filters to apply should be simply just uncommenting defines from the control
 * center below.
 */
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "cannyEdge.h"
#include "houghLine.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;


/** control center **/
#define DEBUG
#define CALC_FPS

#define RES_480
//#define SHOW_RAW
#define APPLY_GAUSSIAN_BLUR
//#define APPLY_HSV_FILTER
//#define APPLY_BOUNDING_BOX
#define APPLY_CANNY_EDGE
//#define APPLY_HOUGH_LINE
/********************/

#ifdef RES_480
const int cam_height = 480;
const int cam_width = 640;
#endif

#ifdef RES_1080
const int cam_height = 1080;
const int cam_width = 1920;
#endif

#ifdef RES_720
const int cam_height = 720;
const int cam_width = 1280;
#endif

//main global variables
Mat cur_frame;
Mat cur_frame_gray;
Mat cur_frame_applied;
Mat gray_edges;
string inputFile;
string trackbarWindow;
string windowName;


/*
 * function to open a VideoCapture with given video filename
 * **unfinished, barely used**
 */
void getVideoFromFile(std::string filename, VideoCapture dest)
{
	if (dest.open(filename) == false) {
		cerr << "Cannot open image file " << filename << endl;
		dest.release();
		return;
	}
	if (dest.isOpened()){
		cout << "Successfully opened " << filename << endl;
	}
}


/*
 * function to set up camera settings before operation
 * **Currently not working. Setting width and height doesn't work**
 */
void cameraSetup(VideoCapture &capture)
{
	//params.filterByColor = true;
	//params.blobColor = 86;

	//properties supported by VideoCapture::set for my setup
	// frame width, height, brightness[0 - 1], contrast, saturation, hue, gain

	//seems like these calls don't do anything? res doesn't change
	capture.set(CV_CAP_PROP_FRAME_WIDTH, cam_width);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, cam_height);
	cout << "Width: " << capture.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
	cout << "Height: " << capture.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
}

/*
 * function to filter/mask orange colors from frame
 * input: 3 channel rgb Mat frame
 * output: 3 channel rgb Mat frame with orange filter mask applied
 * description:
 *  - converts the original frame in HSV color format
 *  - applies hard-coded thresholds to each channel
 *  - masks original image with threshold mask
 */
Mat filterOrange(Mat frame)
{
	GpuMat gpuFrame(frame);
	GpuMat convertedHue;
	gpu::cvtColor(gpuFrame, convertedHue, CV_RGB2HSV);

	vector<GpuMat> hsv_split;
	gpu::split(convertedHue, hsv_split);

	GpuMat hue, sat, val;
	hsv_split[0].copyTo(hue);
	hsv_split[1].copyTo(sat);
	hsv_split[2].copyTo(val);

	int min_hue = 100;
	int max_hue = 120;
	int min_sat = 100;
	int max_sat = 255;
	int min_val = 40;
	int max_val = 255;

	//apply threshold to each channel
	GpuMat hue1, hue2, sat1, sat2, val1, val2, binaries;
	gpu::threshold(hsv_split[0], hue1, min_hue, 179, THRESH_BINARY);
	gpu::threshold(hsv_split[0], hue2, max_hue, 179, THRESH_BINARY_INV);
	gpu::bitwise_and(hue1, hue2, hsv_split[0]);
	gpu::threshold(hsv_split[1], sat1, min_sat, 255, THRESH_BINARY);
	gpu::threshold(hsv_split[1], sat2, max_sat, 255, THRESH_BINARY_INV);
	gpu::bitwise_and(sat1, sat2, hsv_split[1]);
	gpu::threshold(hsv_split[2], val1, min_sat, 255, THRESH_BINARY);
	gpu::threshold(hsv_split[2], val2, max_sat, 255, THRESH_BINARY_INV);
	gpu::bitwise_and(val1, val2, hsv_split[2]);
	//combine binaries
	gpu::bitwise_and(hsv_split[0], hsv_split[1], binaries);
	gpu::bitwise_and(hsv_split[2], binaries, binaries);

	//mask original image with binary
	GpuMat finalGpu;
	gpuFrame.copyTo(finalGpu, binaries);
	//binaries.copyTo(finalGpu);

	Mat final;
	finalGpu.download(final);
	return final;
}

//debugging function to show frame on separate test window.
void show(Mat frame)
{
	namedWindow("test", WINDOW_NORMAL);
	resizeWindow("test", cam_width, cam_height);
	imshow("test", frame);
}

/*
 * function to get binary image of frame
 * input: 3 channel rgb Mat frame
 * output: binary Mat image
 * description:
 *  - Assumes frame has some zero valued pixels. Applies
 *    threshold value of 1 to every pixel.
 *  - Any pixel with value >= 1 is set to 1
 *  - Zero value pixels (black pixels / rgb(0,0,0)) set to 0
 */
Mat getBinary(Mat frame)
{
	GpuMat gpuFrame(frame);
	GpuMat grayFrame;
	GpuMat binary;
	int thresh = 1;
	int maxThresh = 255;

	gpu::cvtColor(gpuFrame, grayFrame, CV_BGR2GRAY);
	gpu::threshold(grayFrame, binary, thresh, maxThresh, THRESH_BINARY);

	Mat final;
	binary.download(final);

	gpuFrame.release();
	grayFrame.release();
	binary.release();

	return final;
}

/*
 * Gaussian blur function
 * input: 3 channel rgb Mat frame
 * output: 3 channel rgb Gaussian blurred Mat frame
 */
Mat applyGaussian(Mat frame)
{
	GpuMat gpuFrame;
	gpuFrame.upload(frame);
	GaussianBlur(gpuFrame, gpuFrame, Size(5,5), 2);
	Mat final;
	gpuFrame.download(final);
	gpuFrame.release();
	return final;
}

struct boundingBox {
	Mat frame;
	float area;
	int centerX;
	int centerY;
	int width;
	int height;
};

/*
 * Function to apply bounding box
 * input: 3 channel rgb Mat frame (blob image)
 * output: 3 channel rgb Mat frame with bounding box applied to biggest blob
 * description:
 *  - converts original frame into binary
 *  - applies a bounding box to the largest blob in the frame (contour with largest area)
 */
Mat applyBoundingBox(Mat frame)
{
	Mat final;
	frame.copyTo(final);
	if (frame.channels() < 3) {
		cerr << "Frame is not 3 channel. Cannot apply bounding box" << endl;
		return final;
	}

	//turn color frame into binary for findContours();
	Mat binary = getBinary(frame);

	//find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(binary, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

 	//Finds the contour with the largest area
	float area = 0;
	float largest = 0;
	int idx;
	for(int i=0; i < contours.size(); i++) {
		area = contourArea(contours[i], false);
		if(area > largest) {
			largest = area;
			idx = i;
		}
	}

	Rect rect;
	rect = boundingRect(contours[idx]);
	Point pt1, pt2;
	pt1.x = rect.x;
	pt1.y = rect.y;
	pt2.x = rect.x + rect.width;
	pt2.y = rect.y + rect.height;

	// Draws the rect in the original image
	rectangle(final, pt1, pt2, CV_RGB(0,0,255), 1);

	return final;
}

/*
 * function containing all transformations used in main infinit loop
 */
Mat applyAll(Mat frame)
{
	Mat final(frame);
#ifdef APPLY_GAUSSIAN_BLUR
	final = applyGaussian(frame);
#endif
#ifdef APPLY_HSV_FILTER
	final = filterOrange(final);
#endif
#ifdef APPLY_BOUNDING_BOX
	final = applyBoundingBox(final);
#endif
#ifdef APPLY_CANNY_EDGE
 	final = applyCannyEdge(final);
#endif
#ifdef APPLY_HOUGH_LINE
 	final = applyHoughLine(final);
#endif
	return final;
}

/*
 * main function
 * description: original design to take in source from either command line or camera feed.
 * - If command argument exists, treat it as a video or image file
 *   - Apply all defined transformations on image or video
 * - Else, source is camera feed. Infinit loop
 */
int main(int argc, char *argv[])
{
	if (argc > 1) { //use files from input command as source instead
		inputFile = argv[1];

		cur_frame = imread(inputFile.c_str(), CV_LOAD_IMAGE_COLOR);
		if (cur_frame.data != NULL) {   //try opening as image
			namedWindow(inputFile.c_str(), WINDOW_NORMAL);
			resizeWindow(inputFile.c_str(), cam_width, cam_height);
			windowName = inputFile;
			trackbarWindow = inputFile + " trackbar";
			namedWindow(trackbarWindow.c_str(), WINDOW_NORMAL);
			cur_frame_applied = applyAll(cur_frame);
		} else {   //try opening as video
			VideoCapture vid;
			getVideoFromFile(inputFile.c_str(), vid);
				//work on this later. not needed now
		}		
		waitKey(0);
		destroyAllWindows();
		exit(0);
	}

	/************************  using camera as source (below)  *********************/
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "Error connecting to a camera device" << endl;
		exit(0);
	}
	windowName = "camera feed - applied";
	namedWindow(windowName, WINDOW_NORMAL);
	resizeWindow(windowName, cam_width, cam_height);
	trackbarWindow = "trackbar";
	namedWindow(trackbarWindow, WINDOW_NORMAL);
	cameraSetup(cap);

	cout << "In capture ..." << endl;
#ifdef CALC_FPS
	string str = "Captures per second: ";
	float cap_per_sec = 0;
	time_t time1 = time(NULL);
	time_t time2 = time(NULL);
	cout << str << "00.00";
	cout.flush();
#endif
	int keyPress;
	while (true) {
		cap.read(cur_frame);
#ifdef SHOW_RAW
		imshow("raw", cur_frame);
#endif
		cur_frame_applied = applyAll(cur_frame);
		if(!cur_frame_applied.empty()){
			imshow(windowName, cur_frame_applied);
		}
		
#ifdef CALC_FPS
		cap_per_sec += 1;
		time2 = time(NULL);
		if (difftime(time2, time1) > 1) {
			time1 = time(NULL);
			cout << string(str.length() + 5, '\b');
			cout << str;
			cout.flush();
			fprintf(stderr, "%2.2f", cap_per_sec);
			cap_per_sec = 0;
		}
#endif
		keyPress = waitKey(1);
		if (keyPress != -1) {
			break;
		}
	}
	return 0;
}
