/*
 * Header file for Hough Transformation
 * Note:
 * - Please set extern variables before using Hough:
 *   - gray_edges: extern single-channel grayscale image set in cannyEdge.cpp
 *   - windowName: name of window which resulting Mat will be shown
 *   - trackbarWindow: name of window in which trackbar will appear in
 */
#ifndef HOUGH_LINE_INCLUDED
#define HOUGH_LINE_INCLUDED
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::gpu;

extern Mat gray_edges;
extern string windowName;
extern string trackbarWindow;

void Probabilistic_Hough(int, void*);
Mat applyHoughLine(Mat frame);
#endif
