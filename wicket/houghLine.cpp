/*
 * File containing the functionality of Hough Transformation
 * Note:
 * -Hough Transformation works in tandem with Canny Edge Transformation.
 *  In other words, the input Mat should be the product of Canny.
 *  The result of canny edge detector produces an image with only
 *  edge contours, perfect as an input for Hough transformation. If
 *  Hough is applied to a regular image, there will be a lot of noise.
 */
#include "houghLine.h"

static GpuMat gpuFrame, hold;
static int min_threshold = 50;
static int max_trackbar = 150;
static int val_trackbar = 75; //starting trackbar val
static Mat hough_final; //final returning frame
static Mat grayFrame;

/*
 * Callback function for hough trackbar
 */
void Probabilistic_Hough( int, void* )
{
}

/*
 * applies Hough Transformation
 * input: three-channel rgb Mat (Canny applied)
 * output: three-channel rgb Mat with Hough lines overlay applied
 */
Mat applyHoughLine(Mat frame)
{
	string houghLabel = "Hough Line Min Threshold";

	createTrackbar( houghLabel.c_str(), trackbarWindow, &val_trackbar, max_trackbar, Probabilistic_Hough);

	gpuFrame.upload(frame);
	gpuFrame.copyTo(hold);
	hold.download(hough_final);
	gpu::cvtColor( gpuFrame, hold, COLOR_BGR2GRAY );
	hold.download(grayFrame);

	// Use Probabilistic Hough Transform
	vector<Vec4i> lines;
	HoughLinesP( grayFrame, lines, 1, CV_PI/180, min_threshold + val_trackbar, 30, 10 );

	// Show the result
	for( size_t i = 0; i < lines.size(); i++ ) {
		Vec4i l = lines[i];
		line( hough_final, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, CV_AA);
	}
	imshow( windowName.c_str(), hough_final );

	//Probabilistic_Hough(0, 0);
	return hough_final;
}
