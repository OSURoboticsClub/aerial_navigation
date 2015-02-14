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

static int min_threshold = 50;
static int max_trackbar = 150;

static Mat hough_final; //final returning frame

static int p_trackbar = 20; //starting trackbar val

void Probabilistic_Hough(int, void*);
Mat applyHoughLine(Mat frame);
#endif
