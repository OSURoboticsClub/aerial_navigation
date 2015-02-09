#include "cannyEdge.h"

void CannyThreshold(int, void*)
{
    /// Reduce noise with a kernel 3x3
	blur( src_gray, result, Size(3,3) );

	/// Canny detector
	Canny( result, result, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	cur_frame.copyTo(dst, result);
	dst.copyTo(cur_frame);
	imshow(inputFile.c_str(), cur_frame);
}

Mat applyCannyEdge(Mat src)
{
    /// Create a matrix of the same type and size as src (for dst)
	dst.create( src.size(), src.type() );
	result.create( src.size(), src.type() );

	/// Convert the image to grayscale
	cvtColor( src, src_gray, CV_BGR2GRAY );

	/// Create a Trackbar for user to enter threshold
	string tmp = inputFile + " trackbar";
	createTrackbar( "Canny Edge Min Threshold:", tmp.c_str(), &lowThreshold, maxThreshold, CannyThreshold );

	/// Show the image
	CannyThreshold(0, 0);
}
