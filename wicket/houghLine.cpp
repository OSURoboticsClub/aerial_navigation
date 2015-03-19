/*
 * File containing the functionality of Hough Transformation
 * Note:
 * -Hough Transformation works in tandem with Canny Edge Transformation.
 *  cannyEdge.h and cannyEdge.cpp is required to use Hough Transformation
 *  because the input of Hough should be a single-channel image such as
 *  an image produced with Canny edge in order to produce meaningful
 *  outputs. If the contrast between neighboring pixels are not distinct,
 *  applying Hough will only produce a noisy output.
 * -gray_edges extern variable in this file stores the image produced
 *  from cannyEdge.cpp file. It is a single-channel grayscale image with
 *  only edge contours.
 */
#include "houghLine.h"

static GpuMat gpuFrame, hold;
static int min_threshold = 50;
static int max_trackbar = 150;
static int val_trackbar = 20; //starting trackbar val
static Mat hough_final; //final returning frame

/*
 * Callback function for hough trackbar
 */
void Probabilistic_Hough( int, void* )
{
}

/*
 * applies Hough Transformation
 * input: three-channel rgb Mat
 * output: three-channel rgb Mat with Hough lines overlay applied
 */
Mat applyHoughLine(Mat frame)
{
	string houghLabel = "Hough Line Min Threshold";

	createTrackbar( houghLabel.c_str(), trackbarWindow, &val_trackbar, max_trackbar, Probabilistic_Hough);

	gpuFrame.upload(gray_edges);
	gpu::cvtColor( gpuFrame, hold, COLOR_GRAY2BGR );
	hold.download(hough_final);

	// Use Probabilistic Hough Transform
	vector<Vec4i> lines;
	HoughLinesP( gray_edges, lines, 1, CV_PI/180, min_threshold + val_trackbar, 30, 10 );

	// Show the result
	for( size_t i = 0; i < lines.size(); i++ ) {
		Vec4i l = lines[i];
		line( hough_final, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, CV_AA);
	}
	imshow( windowName.c_str(), hough_final );

	//Probabilistic_Hough(0, 0);
	return hough_final;
}
