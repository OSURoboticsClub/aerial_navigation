#include "cannyEdge.h"

void CannyThreshold(int, void*)
{
#ifndef APPLY_GAUSSIAN_BLUR
    /// Reduce noise with a kernel 3x3
	gpuFrame.upload(gray_frame);
	gpu::blur( gpuFrame, hold, Size(3,3) );
	hold.download(result);
#else
	gray_frame.copyTo(result);
#endif

	/// Canny detector
	Canny( result, result, lowThreshold, lowThreshold*ratio, kernel_size);

	//store result in its current stage in case hough lines is applied after
	result.copyTo(gray_edges);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);
	original.copyTo(dst, result); //mask with result
	dst.copyTo(final);
	imshow(windowName.c_str(), final);
}

Mat applyCannyEdge(Mat src)
{
	src.copyTo(original); //could be gpu optimized?

    /// Create a matrix of the same type and size as src (for dst)
	dst.create( src.size(), src.type() );

	/// Convert the image to grayscale
	gpuFrame.upload(src);
	gpu::cvtColor( gpuFrame, hold, CV_BGR2GRAY );
	hold.download(gray_frame);

	/// Create a Trackbar for user to enter threshold
	createTrackbar( "Canny Edge Min Threshold:", trackbarWindow.c_str(), &lowThreshold, maxThreshold, CannyThreshold );

	/// Show the image
	CannyThreshold(0, 0);
	return final;
}
