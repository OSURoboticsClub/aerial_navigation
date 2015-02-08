#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <iostream>
#include <stdio.h>
#include <string.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

#define RES_720
#define SHOW_RAW
#define APPLY_CANNY_EDGE
//#define APPLY_SOBEL_DERIV
//#define APPLY_OPENING

#ifdef RES_1080
const int cam_height = 1080;
const int cam_width = 1920;
#endif

#ifdef RES_720
const int cam_height = 720;
const int cam_width = 1280;
#endif

//global variables
Mat cur_frame;
string inputFile;
string trackbarWindow;

void getVideoFromFile(std::string filename, VideoCapture dest)
{
	if (dest.open(filename) == false) {
		cerr << "Cannot open image file " << filename << endl;
		dest.release();
		return;
	}
	if (dest.isOpened()){
		cout << "Successfully opened " << filename << endl;
	}
}

void cameraSetup(VideoCapture &capture)
{
	//params.filterByColor = true;
	//params.blobColor = 86;

	//properties supported by VideoCapture::set for my setup
	// frame width, height, brightness[0 - 1], contrast, saturation, hue, gain

	//seems like these calls don't do anything? res doesn't change
	capture.set(CV_CAP_PROP_FRAME_WIDTH, cam_width);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, cam_height);
	cout << "Width: " << capture.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
	cout << "Height: " << capture.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
}

void wicketOverlay(Mat frame, Point p1, Point p2)
{
	
}

//canny edge globals
Mat canny_src_gray;
Mat canny_dst, canny_result;

int canny_edgeThresh = 1;
int canny_lowThresh;
int const canny_maxThresh = 100;
int canny_ratio = 3;
int canny_kernel_size = 3;

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
	/// Reduce noise with a kernel 3x3
	blur( canny_src_gray, canny_result, Size(3,3) );

	/// Canny detector
	Canny( canny_result, canny_result, canny_lowThresh, canny_lowThresh*canny_ratio, canny_kernel_size );

	/// Using Canny's output as a mask, we display our result
	canny_dst = Scalar::all(0);

	cur_frame.copyTo(canny_dst, canny_result);
	canny_dst.copyTo(cur_frame);
	imshow(inputFile.c_str(), cur_frame);
}

Mat applyCannyEdge(Mat src)
{
	/// Create a matrix of the same type and size as src (for dst)
	canny_dst.create( src.size(), src.type() );
	canny_result.create( src.size(), src.type() );

	/// Convert the image to grayscale
	cvtColor( src, canny_src_gray, CV_BGR2GRAY );

	/// Create a Trackbar for user to enter threshold
	createTrackbar( "Canny Edge Min Threshold:", trackbarWindow.c_str(), &canny_lowThresh, canny_maxThresh, CannyThreshold );

	/// Show the image
	CannyThreshold(0, 0);
}

Mat applySobelDerivative(Mat src)
{
	Mat src_gray;
	Mat grad;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	int c;

	//if( !src.data )
	//{ return NULL; }

	GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

	/// Convert it to gray
	cvtColor( src, src_gray, CV_RGB2GRAY );

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	/// Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	return grad;
}

Mat applyOpening(Mat src)
{
	Mat dst;
	int morph_elem = 0;
	int morph_size = 1;
	int morph_operator = 0;
	int const max_operator = 4;
	int const max_elem = 2;
	int const max_kernel_size = 21;
	//not quite sure how this function works yet
	int operation = morph_operator + 2; //+2 MORPH_OPEN, +3 MORPH_CLOSE, +4 MORPH_TOPHAT, +5 MORPH_BLACKHAT
	Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

	//apply opening morphology
	morphologyEx(src, dst, operation, element);
	return dst;
}

void findWicket(Mat frame)
{
	
}

void processFiles()
{

	exit(0);
}

int main(int argc, char *argv[])
{
	if (argc > 1) { //use files from input command as source instead
		inputFile = argv[1];

		cur_frame = imread(inputFile.c_str(), CV_LOAD_IMAGE_COLOR);
		if (cur_frame.data != NULL) {   //try opening as image
			namedWindow(inputFile.c_str(), WINDOW_NORMAL);
			resizeWindow(inputFile.c_str(), cam_width, cam_height);
			trackbarWindow = inputFile + " trackbar";
			namedWindow(trackbarWindow.c_str(), WINDOW_NORMAL);
#ifdef APPLY_CANNY_EDGE
			applyCannyEdge(cur_frame);
			cout << "Canny Edge applied" << endl;
#endif
#ifdef APPLY_SOBEL_DERIV
			cur_frame = applySobelDerivative(cur_frame);
			cout << "Sobel derivative applied" << endl;
#endif
#ifdef APPLY_OPENING
			cur_frame = applyOpening(cur_frame);
			cout << "Opening morphology applied" << endl;
#endif

		} else {   //try opening as video
			VideoCapture vid;
			getVideoFromFile(inputFile.c_str(), vid);
				//work on this later. not needed now
		}
		
		//apply filter stuff
		//work on this later... not needed now
		waitKey(0);
		exit(0);
	}

	//using camera as source
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "Error connecting to a camera device" << endl;
		exit(0);
	}
	cameraSetup(cap);
	while (true) {
		cap.read(cur_frame);
#ifdef SHOW_RAW
		imshow("raw", cur_frame);
#endif
		findWicket(cur_frame);
		//imshow();
		waitKey(10);
	}
	
	
	return 0;
}