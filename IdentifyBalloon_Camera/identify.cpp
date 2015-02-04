#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <iostream>
#include <stdio.h>
#include <string.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

//define
#define SHOWVIDEO
#define RES_720
#define ENABLE_HSV  //enable/disable for viewing
#define ENABLE_HSV_GUI
//#define ENABLE_RGB_SPLIT

//global variables
Mat frame;
Mat hsvFrame;
int hue_min = 0, sat_min = 0, val_min = 0;
int hue_max = 180, sat_max = 255, val_max = 255;
SimpleBlobDetector::Params params;
//capture = cvCaptureFromCAM( CV_CAP_ANY ); //0=default, -1=any camera, 1..99=your camera


#ifdef RES_1080
const int cam_height = 1080;
const int cam_width = 1920;
#endif

#ifdef RES_720
const int cam_height = 720;
const int cam_width = 1280;
#endif

#ifdef SHOWVIDEO
void initVideo(VideoCapture &capture)
{
	namedWindow("raw");
#ifdef ENABLE_HSV
	namedWindow("hsv");
	capture.read(frame);
	moveWindow("hsv", 640, 0);
	imshow("hsv", frame);
	moveWindow("hsv", 640, 0);
	//namedWindow("val");
#endif
#ifdef ENABLE_RGB_SPLIT
	namedWindow("Using RGB Channel Split");
#endif
}
#endif
	
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

void hsvOnChange(int val, void *data)
{
	cout << val << endl;
	int &key = *(static_cast<int*>(data));
	switch(key) {
	case 0:
		hue_min = val;
		break;
	case 1:
		sat_min = val;
		break;
	case 2:
		val_min = val;
		break;
	case 3:
		hue_max = val;
		break;
	case 4:
		sat_max = val;
		break;
	case 5:
		val_max = val;
		break;
	default:
		cout << "value not captured" << endl;	   
	}

}

void hsv_gui_init()
{
	int key = 0;
	int tmp = 0;
	createTrackbar("min hue", "hsv", 0, 180, hsvOnChange, static_cast<void*>(&key));
	key = 1;
	createTrackbar("min sat", "hsv", 0, 255, hsvOnChange, static_cast<void*>(&key));
	key = 2;
	createTrackbar("min val", "hsv", 0, 255, hsvOnChange, static_cast<void*>(&key));
	key = 3;
	tmp = 180;
	createTrackbar("max hue", "hsv", &tmp, 180, hsvOnChange, static_cast<void*>(&key));
	key = 4;
	tmp = 255;
	createTrackbar("max sat", "hsv", &tmp, 255, hsvOnChange, static_cast<void*>(&key));
	key = 5;
	createTrackbar("max val", "hsv", &tmp, 255, hsvOnChange, static_cast<void*>(&key));
}

#ifdef ENABLE_RGB_SPLIT
void redChannelSplit(Mat frame)
{
	vector<Mat> rgb_split;
	split(frame, rgb_split);
	
	GpuMat red_channel(rgb_split[2]);
	GpuMat gpu_frame(frame);
	GpuMat gpu_frame_gray;
	gpu::cvtColor(gpu_frame, gpu_frame_gray, CV_BGR2GRAY);
	GpuMat redFrame;
	absdiff(red_channel, gpu_frame_gray, redFrame);
	Mat final;
	redFrame.download(final);
	namedWindow("Using RGB Channel Split");
	imshow("Using RGB Channel Split", final);	
}
#endif

/*
  Note:
  seems like inRange only takes in Mat parameters, not GpuMat
 */

void renderHSV(Mat frame)
{
	//Mat -> GpuMat
    GpuMat GpuHue(frame);
	//RGB -> HSV
	GpuMat convertedHue;
	gpu::cvtColor(GpuHue, convertedHue, CV_RGB2HSV);

	//Filter red
	Mat redHueFrame;
#ifdef ENABLE_HSV_GUI
	//********PROBLEM: can't use inRange because not rgb anymore?********//
	Mat low_bound;
	low_bound = cv::Scalar(0,0,0);
	Mat up_bound;
	up_bound = cv::Scalar(255,100,100);
	GpuMat g_low(low_bound);
	GpuMat g_up(up_bound);
	Mat tmp;
	convertedHue.download(tmp);
	inRange(tmp, Scalar(hue_min,sat_min,val_min), Scalar(hue_max,sat_max,val_max), redHueFrame);
#else
	//hard coded lower and upper bound for red
	inRange(convertedHue, Scalar(0, 0, 0), Scalar(255, 100, 100), redHueFrame);
#endif
	//convertedHue.download(hsvFrame);
	hsvFrame = redHueFrame;
}


// std::tuple findBlob(Mat frame)
// {
//     blob_detector.detect(frame, keypoints);
// 	// extract the x y coordinates of the keypoints:
// 	for (int i=0; i<keypoints.size(); i++){
// 		float x = keypoints[i].pt.x;
// 		float y = keypoints[i].pt.y;
// 		cout << x << " " << y << endl;
// 	}

// 	return std::make_tuple(x, y);
// }

int main( int argc, const char** argv )
{
	VideoCapture capture(CV_CAP_ANY);
	if (capture.isOpened()) {
		cameraSetup(capture);
#ifdef SHOWVIDEO
		initVideo(capture);
#endif
#ifdef ENABLE_HSV_GUI
		hsv_gui_init();
#endif
		//cv::SimpleBlobDetector blob_detector(params);
		
		vector<cv::KeyPoint> keypoints;
		cout << "In capture ..." << endl;
		string str = "Captures per second: ";
		float cap_per_sec = 0;
		time_t time1 = time(NULL);
		time_t time2 = time(NULL);
		cout << str << "00.00";
		cout.flush();
		for(;;) {
			//capture frame from camera
			capture.read(frame);
			if( frame.empty() ){
				break;
			}
			//render HSV from captured frame
			renderHSV(frame);
#ifdef ENABLE_RGB_SPLIT
			redChannelSplit(frame);
#endif
			//findBlob(frame);
			
#ifdef SHOWVIDEO
			imshow("raw", frame);	
#ifdef ENABLE_HSV
			imshow("hsv", hsvFrame);
			//imshow("val", valFrame);
#endif
			waitKey(5);
#endif
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
		}
	} else {
		cout << "no camera detected" << endl;
		exit(-1);
	}
	printf("\n");
	
	waitKey(0);
	
#ifdef SHOWVIDEO	
	cvDestroyWindow("raw");
#endif		
	return 0;
}
