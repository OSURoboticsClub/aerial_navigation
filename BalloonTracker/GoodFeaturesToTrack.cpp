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
using namespace cv::gpu;
using namespace std;

Mat image, frame, gray, mask;
gpu::GpuMat gpuimage, gpuframe, gpugray, gpumask, prev_points, next_points;

Rect selection;
Point origin;
bool selectObject = false;
int trackObject = 0;

RNG rng(12345);

static void onMouse( int event, int x, int y, int, void* )
{
	if( selectObject )
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);

		selection &= Rect(0, 0, image.cols, image.rows);
	}

	switch( event )
	{
	case CV_EVENT_LBUTTONDOWN:
		origin = Point(x,y);
		selection = Rect(x,y,0,0);
		selectObject = true;
		break;
	case CV_EVENT_LBUTTONUP:
		selectObject = false;
		if( selection.width > 0 && selection.height > 0 )
			trackObject = -1;
		break;
	}
}
static void download(const GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

static void download(const GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}
int main( int argc, const char** argv )
{

	VideoCapture cap;
	Rect trackWindow;

	cap.open("/home/scott/Aerial/aerial_navigation/photos/SoccerGoal.MOV");

	if( !cap.isOpened() )
	{
		cout << "***Could not initialize capturing...***\n";
		return -1;
	}
	//GoodFeaturesToTrackDetector_GPU(maxCorners=1000, qualityLevel=0.01, minDistance=0.0, blockSize=3, useHarrisDetector=false, harrisK=0.04)
	gpu::GoodFeaturesToTrackDetector_GPU good(4, 0.2, 1.0, 3, true, 0.04);

	namedWindow( "Good Features to Track Detector", 0 );
	setMouseCallback( "Good Features to Track Detector", onMouse, 0 );

	bool paused = false;
	cap >> frame;
	paused = true;
	vector<Point2f> prevPts;
	for(;;)
	{
		if( !paused )
		{
			cap >> frame;
			if( frame.empty() )
				break;
		}

		frame.copyTo(image);

		if( !paused )
		{
			gpuframe.upload(frame);
			gpu::cvtColor(gpuframe, gpugray, COLOR_BGR2GRAY);

			if(trackObject < 0) {

				trackObject = 1;
			}
			if(trackObject){
				GpuMat gpumask(gpugray.size(), CV_8UC1, Scalar::all(0));
				gpumask(selection).setTo(Scalar::all(255));

				gpu::Laplacian(gpugray, gpugray, gpugray.depth(), 3, 1.0, BORDER_DEFAULT);
				//gpugray.download(gray);
				//imshow("lap",gray);
				good(gpugray, prev_points, gpumask);
			} else {
				good(gpugray, prev_points);
			}
			prevPts = vector<Point2f>(prev_points.cols);
			download(prev_points, prevPts);
		}

		for(int i = 0; i < prevPts.size(); i++){
			circle(image, prevPts[i], 4, Scalar(225, 60, 60), -1, 8, 0 );
		}
		if( trackObject < 0 )
			paused = false;
		if( selectObject && selection.width > 0 && selection.height > 0 )
		{
			Mat mask(image, selection);
			bitwise_not(mask, mask);
		}
		imshow( "Good Features to Track Detector", image );

		char c = (char)waitKey(10);
		if( c == 27 )
			break;
		switch(c)
		{
		case 'c':
		trackObject = 0;
		break;
		case 'p':
		paused = !paused;
		break;
		default:
			;
		}
	}

	return 0;
}
