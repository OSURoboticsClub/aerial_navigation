#ifndef HOUGH_LINE_INCLUDED
#define HOUGH_LINE_INCLUDED
#include <opencv2/imgproc/imgproc.hpp>
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

//static Mat edges; //testing, edges = cur_frame
//static Mat standard_hough;
static Mat probabilistic_hough;
static int min_threshold = 50;
static int max_trackbar = 150;

//static int s_trackbar = max_trackbar;
static int p_trackbar = 0; //starting trackbar val

void Probabilistic_Hough(int, void*);
void applyHoughLine(Mat frame);
#endif
