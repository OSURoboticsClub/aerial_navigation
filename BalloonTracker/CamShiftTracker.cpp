/*
 * This program implements the camshift tracking algorithm.
 * It is capable of tracking a selected object based on hue.
 * Also it has the ability to filter out the pixels that aren't
 * within saturation and value ranges set by trackbars.
 *
 * This was designed to track a balloon for the Sparkfun AVC competition
 * should work for any object identifiable by color. The tracking area
 * will adapt as the object moves around in the image. However this
 * method can fail when it gets put near an object that is similar in color.
 *
 * NOTE: this is only partially implemented on the GPU. OpenCV doesn't have
 * a gpu version. However there are ways to get in on the gpu. This does
 * work quite quickly as is for 640x480. But to do at 720 or 1080 it might be
 * Necessary to implement the rest.
 *
 * There is also a simple 2D kalman filter implemented here to try and help
 * improve reliably tracking the object with occlusions and other objects
 * that are similar.
 *
 * To use the program run it. Two windows will pop up. One from a camera feed
 * and the other is for displaying the histogram of colors that it is looking
 * for in the image. To being tracking simply drag over the object you want to
 * track in the first window. It is a good idea to make sure the rectangle is
 * completely inside the object.
 *
 * To see how the filter changes the image press 'b' then adjust the track bars
 * so the object is only selected if possible. This will increase tracking
 * performance.
 *
 * To turn on the kalman filer press 'k'. Pressing 'k' will also show the
 * computation time for different parts of the algorithm along with the frame
 * rate.
 *
 */
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <ctype.h>
#include <vector>
#include <sys/time.h>
#include <stdio.h>

using namespace cv;
using namespace std;

//values used for changing filter parameters
int s_min = 0; //Saturation minimum
int v_min = 0; //value minimum
int v_max = 256; //value maximum

//name of trackbar window
const string trackbarWindowName = "Trackbars";
//used to have universal access to the image passed to the camera
Mat image;

//state variables
bool backprojMode = false; //turns on and off display of filters
bool selectObject = false; //used as a state variable to know if an object has been selected
int trackObject = 0; //used to know if we should be tracking an object
bool showHist = true; //used to turn on and off histogram display


vector<Point> mousev,kalmanv; //list of points for kalman prediction and object measured location. Front is first point back is most recent
Point origin; //first point used in selecting the object
Rect selection; //the rectangle of defining the selection

static void onMouse( int event, int x, int y, int, void* )
{
	if( selectObject ) //we have started selecting the object
	{
		//set top left point
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		//set dimensions
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);

		selection &= Rect(0, 0, image.cols, image.rows);
	}

	switch( event )
	{
	case CV_EVENT_LBUTTONDOWN:
		origin = Point(x,y); //set the first point in the selection
		selection = Rect(x,y,0,0);
		selectObject = true; //change state to selecting
		break;
	case CV_EVENT_LBUTTONUP:
		selectObject = false; //no longer selecting
		if( selection.width > 0 && selection.height > 0 ) //if the selection is a good rectangle
			trackObject = -1; //set tracking to be in the first state
		break;
	}
}
void on_trackbar( int, void* )
{
	//This function gets called whenever a
	// trackbar position is changed
	//no need to have anything in here
}
void createTrackbars(){
	//create window for trackbars
    namedWindow(trackbarWindowName,0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf( TrackbarName, "s_min", s_min);
	sprintf( TrackbarName, "v_min", v_min);
	sprintf( TrackbarName, "v_max", v_max);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.v_max),
	//the max value the trackbar can move (eg. v_max),
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->
    createTrackbar( "s_min", trackbarWindowName, &s_min, 255, on_trackbar );
    createTrackbar( "v_min", trackbarWindowName, &v_min, v_max, on_trackbar );
    createTrackbar( "v_max", trackbarWindowName, &v_max, v_max, on_trackbar );



}
static void help()
{
	cout << "\nThis is a demo that shows mean-shift based tracking\n"
			"You select a color objects such as your face and it tracks it.\n"
			"This reads from video camera (0 by default, or the camera number the user enters\n"
			"Usage: \n"
			"   ./camshiftdemo [camera number]\n";

	cout << "\n\nHot keys: \n"
			"\tESC - quit the program\n"
			"\tc - stop the tracking\n"
			"\tb - switch to/from backprojection view\n"
			"\th - show/hide object histogram\n"
			"\tk - start/stop using kalmanfilter. Also shows computation time"
			"\tp - pause video\n"
			"To initialize tracking, select the object with mouse\n";
}
/*
 * kalman_init
 * Inputs:
 * 		kal -> reference of a kalman filter object that is not null
 * 		p  	-> starting point of the object
 * 		processNoiseCov -> value of the noise that can happen during processing
 * 		measurementNoiseCov -> value representing noise of the measurement. Make this value smaller if the cv algorithm can accurately identify the location
 * 		errorCovPost -> value of the error somewhere kalman filter so tuning is required
 */
void kalman_init(KalmanFilter &kal, Point p, double processNoiseCov, double measurementNoiseCov, double errorCovPost){
	kal.statePre.at<float>(0) = p.x; //starting x point
	kal.statePre.at<float>(1) = p.y; //starting y point
	kal.statePre.at<float>(2) = 0;   //this is the x velocity
	kal.statePre.at<float>(3) = 0;   //this is the y velocity
	//simple transition matrix. Look up online how to make one if you need to change this when adding more variables than these 4
	kal.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);
	//set all the values in the identity
	setIdentity(kal.measurementMatrix);
	setIdentity(kal.processNoiseCov, Scalar::all(processNoiseCov));
	setIdentity(kal.measurementNoiseCov, Scalar::all(measurementNoiseCov));
	setIdentity(kal.errorCovPost, Scalar::all(errorCovPost));
}
/*
 * used to calculate the difference in time measurements
 */
long getTimeDelta(struct timeval timea, struct timeval timeb) {
	return 1000000 * (timeb.tv_sec - timea.tv_sec) +
			(int(timeb.tv_usec) - int(timea.tv_usec));
}


int main( int argc, const char** argv )
{
	help();
	//time values, time a,b are used for the overall timing, S,E are used for local time calculations
	struct timeval timea, timeb, timeS, timeE;
	//used to store the amount of time for each section of the program
	//totalTime = total time one pass takes
	//camTime = time spent doing camShift
	//kalTime = time spent doing kalaman filtering
	long totalTime = 0, camTime = 0, kalTime = 0;
	//number of frames processed
	int nFrames = 0;


	KalmanFilter KF(4, 2, 0); //Initial setting for the kalman filter
	Mat_<float> state(4, 1); /* (x, y, Vx, Vy) */
	Mat processNoise(4, 1, CV_32F); //noise matrix
	Mat_<float> measurement(2,1); measurement.setTo(Scalar(0)); //measurement values
	Point pt(0, 0); //initial point used
	mousev.push_back(pt); //list for measurement points
	kalmanv.push_back(pt); //list for kalman filter suggest points after correction

	//state variables
	bool kalman = false; //turning kalman filtering on/off
	bool paused = false; //toggling pause state

	VideoCapture cap; //capturing video off of camera
	Rect trackWindow;
	int hsize = 32; //number of color bins for histogram. Usually 16. lowering this increase speed slightly
	float hranges[] = {0,180};  //value ranges for hue values
	const float* phranges = hranges;

	cap.open(0); //open camera to default camera

	if( !cap.isOpened() ) //make sure camera is open
	{
		help();
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";
		return -1;
	}

	//check frame resolution and try and set it (currently doesn't work on Jetson)
	cout << ": width=" << cap.get(CV_CAP_PROP_FRAME_WIDTH) << ", height=" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
//	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280 );
//	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720 );
//	cout << ": width=" << cap.get(CV_CAP_PROP_FRAME_WIDTH) << ", height=" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;

	namedWindow( "Histogram", 0 ); //histogram window
	namedWindow( "CamShift Demo", 0 ); //video feed window
	setMouseCallback( "CamShift Demo", onMouse, 0 ); //set this so trackbars update values
	createTrackbars(); //creates trackbars

	Mat frame, hsv, hue, mask, thresh, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
	gpu::GpuMat gpuf, gpuhsv, gpumask, gpuThresh1, gpuThresh2;
	vector<gpu::GpuMat> hsvplanes(3);

	cap >> frame;
	if( frame.empty() )
		return 0;
	paused = false;
	for(;;)
	{
		gettimeofday(&timea, NULL); //start timing the total iteration
		if( !paused )
		{
			cap >> frame; //pull frame off of the camera
			nFrames++; //increment frame count
			if( frame.empty() )
				break;
		}

		frame.copyTo(image); //copy the frame. This is only used to display the frame

		if( !paused ){
			//convert bgr image to HSV values
			//cvtColor(image, hsv, COLOR_BGR2HSV); //no gpu color conversion
			gpuf.upload(frame); //upload current image to gpu memory
			gpu::cvtColor(gpuf, gpuhsv, COLOR_BGR2HSV); //convert to HSV on gpu
			gpuhsv.download(hsv); //download HSV image to local cpu memory
			if( trackObject )
			{

				int _vmin = v_min, _vmax = v_max;
				gettimeofday(&timeS, NULL); //start timing for camshift (should consider breaking this time out and including it with other conversion time)

				//create a mask to use only the values that are in the filter ranges
				//inRange(hsv, Scalar(0, s_min, MIN(_vmin,_vmax)), Scalar(180, 256, MAX(_vmin, _vmax)), mask); //non gpu version

				//gpu version of inRange (not sure how much faster this is, but should be noticeable)
				gpu::split(gpuhsv,hsvplanes); //split the image into hue saturation and value
				hsvplanes[1].setTo(Scalar(255)); //mark all values in HUE to be on
				gpu::threshold(hsvplanes[2], hsvplanes[2], s_min, 255, THRESH_BINARY); //allow only pixels with saturation value >= s_min
				gpu::threshold(hsvplanes[3], gpuThresh1, _vmin, 255, THRESH_BINARY); //get all pixels that have a val >= _vmin
				gpu::threshold(hsvplanes[3], gpuThresh2, _vmax, 255, THRESH_BINARY_INV); //get all pixels that have a val <= _vmax
				gpu::bitwise_and(gpuThresh1, gpuThresh2, hsvplanes[3]); //get only the pixels that have val in the range [_vmin, _vmax]
				gpu::merge(hsvplanes, 3, gpumask); //combine the individual matrices into one HSV image
				gpumask.download(hsv); //download HSV image to local cpu memory

				int ch[] = {0, 0};
				hue.create(hsv.size(), hsv.depth());
				mixChannels(&hsv, 1, &hue, 1, ch, 1);

				gettimeofday(&timeE, NULL); //stop timing camshift
				camTime += getTimeDelta(timeS, timeE); //add the time to total camshift time

				if( trackObject < 0 ) //if we haven't already calculated the histogram of colors (first pass through after selecting object
				{
					if(kalman){ //if using kalman tracking
						gettimeofday(&timeS, NULL); //start timing kalman
						kalman_init(KF, mousev.back(), 1e-4, 1e-1, .1);

						mousev.clear(); //clear measured points
						kalmanv.clear(); //clear kalman points

						gettimeofday(&timeE, NULL); //start timing kalman
						kalTime += getTimeDelta(timeS, timeE);

					}
					Mat roi(hue, selection), maskroi(mask, selection); //define region of interest (the area we selected)

					//this might be able to be done on the GPU, but it hasn't been tested for this application.
					calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges); //calculate the histogram of colors
					normalize(hist, hist, 0, 255, CV_MINMAX);

					trackWindow = selection;
					trackObject = 1;

					//build the image for displaying the histogram
					histimg = Scalar::all(0);
					int binW = histimg.cols / hsize;
					Mat buf(1, hsize, CV_8UC3);
					for( int i = 0; i < hsize; i++ )
						buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
					cvtColor(buf, buf, CV_HSV2BGR);

					for( int i = 0; i < hsize; i++ )
					{
						int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
						rectangle( histimg, Point(i*binW,histimg.rows),
								Point((i+1)*binW,histimg.rows - val),
								Scalar(buf.at<Vec3b>(i)), -1, 8 );
					}

				}
				gettimeofday(&timeS, NULL); //start timing for camshift
				calcBackProject(&hue, 1, 0, hist, backproj, &phranges); //calculate back projection of the histogram on the hue channel
				backproj &= mask; //apply mask
				//use camshift alg to find the object with a rectangle roated to fit the object orientation
				RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
				gettimeofday(&timeE, NULL); //stop timing for camshift
				camTime += getTimeDelta(timeS, timeE); //add time to camshift total time

				if(kalman){
					gettimeofday(&timeS, NULL); //start timing for kalman filter
					Mat prediction = KF.predict();	//predict where we think the object will be (could use this prediction to mask an image for doing camshift as a way to speed up processing. At this time not nessicary)
					Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
					//set the location of the measurement point
					measurement(0) = trackBox.center.x;
					measurement(1) = trackBox.center.y;

					Point measPt(measurement(0),measurement(1));
					mousev.push_back(measPt);

					Mat estimated = KF.correct(measurement);//correct the prediction with the measurement
					Point statePt(estimated.at<float>(0),estimated.at<float>(1)); //get the new estimated point
					kalmanv.push_back(statePt);
					//this is function that draws a cross at a given point
#define drawCross( center, color, d )                     \
line( image, Point( center.x - d, center.y - d ),           \
Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); \
line( image, Point( center.x + d, center.y - d ),           \
Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )

					drawCross( statePt, Scalar(255,0,0), 5 ); //draw cross for the state point (blue)
					drawCross( measPt, Scalar(0,0,255), 5 );  //draw cross for the measurement point (red)

					gettimeofday(&timeE, NULL); //stop timing for kalman filter
					kalTime += getTimeDelta(timeS, timeE);

				}
				if( trackWindow.area() <= 1 )
				{
					int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
					trackWindow = Rect(	trackWindow.x - r, trackWindow.y - r,
										trackWindow.x + r, trackWindow.y + r) &
								  Rect(0, 0, cols, rows);
				}

				if( backprojMode ) {//if showing the filtered image convert it to gray scale
					cvtColor( backproj, image, COLOR_GRAY2BGR );
				}
				ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA ); //draw ellipse around the object.
			}
		}
		else if( trackObject < 0 )
			paused = false;

		if( selectObject && selection.width > 0 && selection.height > 0 ) //update the image to show what area is being selected
		{
			Mat roi(image, selection);
			bitwise_not(roi, roi);
		}
		gettimeofday(&timeb, NULL); //end total time accumulation
		totalTime += getTimeDelta(timea, timeb);
		imshow( "CamShift Demo", image );
		imshow( "Histogram", histimg );

		char c = (char)waitKey(10);
		//user interaction
		if( c == 27 )
			break;
		switch(c)
		{
		case 'b':
			backprojMode = !backprojMode;
			break;
		case 'c':
			trackObject = 0;
			histimg = Scalar::all(0);
			break;
		case 'h':
			showHist = !showHist;
			if( !showHist )
				destroyWindow( "Histogram" );
			else
				namedWindow( "Histogram", 1 );
			break;
		case 'p':
			paused = !paused;
			break;
		case 'k':
			kalman = !kalman;
			cout << "frames                       : " << nFrames << endl;
			cout << "TotalTime                    : " << double(totalTime)/1000000.0 << endl;
			cout << "FPS                          : " << double(nFrames)/(double(totalTime)/1000000.0) << endl;
			cout << "Percentage CamShift Time     : " << double(camTime)/double(totalTime) << endl;
			cout << "Percentage KalmanFilter Time : " << double(kalTime)/double(totalTime) << endl;
			totalTime = camTime = kalTime = 0;
			nFrames = 0;
			break;
		default:
			;
		}
	}

	return 0;
}
