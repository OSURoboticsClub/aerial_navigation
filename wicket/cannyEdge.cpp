#include "cannyEdge.h"

void CannyThreshold(int, void*)
{
#ifndef APPLY_GAUSSIAN_BLUR
    /// Reduce noise with a kernel 3x3
	blur( cur_frame_gray, result, Size(3,3) );
#else
	cur_frame_gray.copyTo(result);
#endif

	/// Canny detector
	Canny( result, result, lowThreshold, lowThreshold*ratio, kernel_size);

	//store result in its current stage in case hough lines is applied after
	result.copyTo(gray_edges);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);
	cur_frame.copyTo(dst, result); //mask with result
	dst.copyTo(cur_frame_applied);
	imshow(windowName.c_str(), cur_frame_applied);
}

void applyCannyEdge(Mat src)
{
    /// Create a matrix of the same type and size as src (for dst)
	dst.create( src.size(), src.type() );

	/// Convert the image to grayscale
	if (cur_frame_gray.empty()){
		cerr << "frame converted to gray in cannyEdge.cpp (No Red Channel Split)" << endl;
		cvtColor( src, cur_frame_gray, CV_BGR2GRAY );
	}

	/// Create a Trackbar for user to enter threshold
	createTrackbar( "Canny Edge Min Threshold:", trackbarWindow.c_str(), &lowThreshold, maxThreshold, CannyThreshold );

	/// Show the image
	CannyThreshold(0, 0);
}
