/*
 * Canny Edge Detector header file
 * Note:
 * Please set extern variables before using this file:
 * - windowName: the window which the resulting Mat will be shown
 * - trackbarWindow: the window which the slider will appear in.
 * - gray_edges: snapshot variable for Hough Transformation
 */

#ifndef CANNY_EDGE_INCLUDED
#define CANNY_EDGE_INCLUDED
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::gpu;

extern Mat gray_edges;
extern string windowName;
extern string trackbarWindow;

void CannyThreshold(int, void*);
Mat applyCannyEdge(Mat src);
#endif
