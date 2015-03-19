/*
 * Canny Edge Detector header file
 * Note:
 * Please set extern variables windowName and trackbarWindow before using
 * this file, windowName being the window which the resulting Mat will be
 * shown and trackbarWindow being the window which the slider will appear
 * in.
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

static Mat dst, result;
static Mat gray_frame;
static Mat original, final;
static GpuMat gpuFrame, hold;

static int lowThreshold = 30; //default threshold
static int const maxThreshold = 100;
static int ratio = 3;
static int kernel_size = 3;

void CannyThreshold(int, void*);
Mat applyCannyEdge(Mat src);
#endif
