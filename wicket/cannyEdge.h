#ifndef CANNY_EDGE_INCLUDED
#define CANNY_EDGE_INCLUDED
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

extern Mat cur_frame;
extern Mat cur_frame_gray;
extern Mat cur_frame_applied;
extern Mat gray_edges;
extern string windowName;
extern string trackbarWindow;

static Mat dst, result;

static int edgeThreshold = 1;
static int lowThreshold = 10; //default threshold
static int const maxThreshold = 100;
static int ratio = 3;
static int kernel_size = 3;

void CannyThreshold(int, void*);
void applyCannyEdge(Mat src);
#endif
