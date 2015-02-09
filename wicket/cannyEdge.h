#ifndef CANNY_EDGE_INCLUDED
#define CANNY_EDGE_INCLUDED
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/gpu/gpu.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <stdio.h>

using namespace std;
using namespace cv;

extern Mat cur_frame;
extern string inputFile;

static Mat src_gray;
static Mat dst, result;

static int edgeThreshold = 1;
static int lowThreshold;
static int const maxThreshold = 100;
static int ratio = 3;
static int kernel_size = 3;

void CannyThreshold(int, void*);
Mat applyCannyEdge(Mat src);
#endif
