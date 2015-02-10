#include "houghLine.h"

void Probabilistic_Hough( int, void* )
{
	vector<Vec4i> p_lines;
	cvtColor( gray_edges, probabilistic_hough, COLOR_GRAY2BGR );

	/// 2. Use Probabilistic Hough Transform
	HoughLinesP( gray_edges, p_lines, 1, CV_PI/180, min_threshold + p_trackbar, 30, 10 );

	/// Show the result
	for( size_t i = 0; i < p_lines.size(); i++ )
		{
			Vec4i l = p_lines[i];
			line( probabilistic_hough, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, CV_AA);
		}

	imshow( windowName.c_str(), probabilistic_hough );
}

void applyHoughLine(Mat frame)
{
	string houghLabel = "Hough Line Min Threshold";

	createTrackbar( houghLabel.c_str(), trackbarWindow, &p_trackbar, max_trackbar, Probabilistic_Hough);

	/// Initialize
	//Standard_Hough(0, 0);
	Probabilistic_Hough(0, 0);
}
