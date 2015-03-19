/*
 * Canny Edge Detector
 * Description:
 * This file contains the functionality for applying canny edge detector
 * on an OpenCV Mat structure.
 * Note:
 * -Some predefined variables are set in the header file cannyEdge.h
 * -It is also preferred that Gaussian blur is applied to the incoming
 * Mat structure before applying canny edge detector to reduce noise. Else,
 * a regular blur is applied instead.
 */
#include "cannyEdge.h"

static Mat dst, result;
static Mat gray_frame;
static Mat original, final;
static GpuMat gpuFrame, hold;

static int lowThreshold = 30; //default threshold
static int const maxThreshold = 100;
static int ratio = 3;
static int kernel_size = 3;

/*
 * Callback function for toolbar
 */
void CannyThreshold(int, void*)
{
}

/*
 * function for applying canny edge
 * input: 3 channel rgb Mat
 * output: 3 channel rgb Mat with canny edge mask applied
 */
Mat applyCannyEdge(Mat src)
{
	src.copyTo(original); //could be gpu optimized?
    /// Create a matrix of the same type and size as src (for dst)
	dst.create( src.size(), src.type() );

	/// Convert the image to single-channel grayscale
	gpuFrame.upload(src);
	gpu::cvtColor( gpuFrame, hold, CV_BGR2GRAY );
	hold.download(gray_frame);

	/// Create a Trackbar for user to enter threshold
	createTrackbar( "Canny Edge Min Threshold:", trackbarWindow.c_str(), &lowThreshold, maxThreshold, CannyThreshold );

#ifndef APPLY_GAUSSIAN_BLUR
    //Increase kernel matrix size for more blur (odd increments) 
	gpuFrame.upload(gray_frame);
	gpu::blur( gpuFrame, hold, Size(3,3) );
	hold.download(result);
#else
	gray_frame.copyTo(result);
#endif
	// Applying canny detector
	Canny( result, result, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);
	original.copyTo(dst, result); //mask original image with canny result
	dst.copyTo(final);
	imshow(windowName.c_str(), final);

	//CannyThreshold(0, 0);
	return final;
}
