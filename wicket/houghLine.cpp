#include "houghLine.h"

void Probabilistic_Hough( int, void* )
{
	GpuMat gpuFrame(gray_edges);
	GpuMat hold;
	cvtColor( gpuFrame, hold, COLOR_GRAY2BGR );
	hold.download(hough_final);
	hold.release();
	gpuFrame.release();

	/// 2. Use Probabilistic Hough Transform
	vector<Vec4i> p_lines;
	HoughLinesP( gray_edges, p_lines, 1, CV_PI/180, min_threshold + p_trackbar, 30, 10 );

	/// Show the result
	for( size_t i = 0; i < p_lines.size(); i++ )
		{
			Vec4i l = p_lines[i];
			line( hough_final, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, CV_AA);
		}

	imshow( windowName.c_str(), hough_final );
}

Mat applyHoughLine(Mat frame)
{
	string houghLabel = "Hough Line Min Threshold";

	createTrackbar( houghLabel.c_str(), trackbarWindow, &p_trackbar, max_trackbar, Probabilistic_Hough);

	/// Initialize
	Probabilistic_Hough(0, 0);
	return hough_final;
}
