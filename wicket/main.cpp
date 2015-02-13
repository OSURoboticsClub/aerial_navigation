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

#define RES_480
#define SHOW_RAW
#define APPLY_HSV_FILTER
#define APPLY_GAUSSIAN_BLUR
#define APPLY_CANNY_EDGE
#define APPLY_HOUGH_LINE

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

void wicketOverlay(Mat frame, Point p1, Point p2)
{
	
}



Mat applyOpening(Mat src)
{
	Mat dst;
	int morph_elem = 0;
	int morph_size = 1;
	int morph_operator = 0;
	int const max_operator = 4;
	int const max_elem = 2;
	int const max_kernel_size = 21;
	//not quite sure how this function works yet
	int operation = morph_operator + 2; //+2 MORPH_OPEN, +3 MORPH_CLOSE, +4 MORPH_TOPHAT, +5 MORPH_BLACKHAT
	Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

	//apply opening morphology
	morphologyEx(src, dst, operation, element);
	return dst;
}

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

	int min_hue = 70;
	int max_hue = 130;
	int min_sat = 70;
	int max_sat = 255;
	int min_val = 150;
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

	GpuMat finalGpu;
	gpuFrame.copyTo(finalGpu, binaries);

	Mat final;
	finalGpu.download(final);
	return final;
}

void show(Mat frame)
{
	namedWindow("test", WINDOW_NORMAL);
	resizeWindow("test", 1280, 720);
	imshow("test", cur_frame_applied);
	waitKey();
	destroyWindow("test");
}

Mat findWicket(Mat frame)
{
	Mat final;
#ifdef APPLY_HSV_FILTER
	final = filterOrange(frame);
#endif
#ifdef APPLY_GAUSSIAN_BLUR
	GpuMat blur;
	blur.upload(final);
	GaussianBlur(blur, blur, Size(9,9), 2);
	blur.download(final);
	blur.release();	
#endif
#ifdef APPLY_CANNY_EDGE
	final = applyCannyEdge(final);
#endif
#ifdef APPLY_HOUGH_LINE
	final = applyHoughLine(final);
#endif
	return final;
}

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
			cur_frame_applied = findWicket(cur_frame);
		} else {   //try opening as video
			VideoCapture vid;
			getVideoFromFile(inputFile.c_str(), vid);
				//work on this later. not needed now
		}		
		waitKey(0);
		destroyAllWindows();
		exit(0);
	}

	//using camera as source
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
	while (true) {
		cap.read(cur_frame);
#ifdef SHOW_RAW
		imshow("raw", cur_frame);
#endif
		cur_frame_applied = findWicket(cur_frame);
		imshow(windowName, cur_frame_applied);
		waitKey(10);
	}
	return 0;
}
