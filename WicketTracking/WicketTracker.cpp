/*
 * This is the program for tracking the corner of the wicket
 * for the Sparkfun AVC. This program uses a soccer goal to simulate
 * the wicket.
 *
 * To use the program you need the WicketTraining photos found on github
 * and the video SoccerGoal2. Make sure they are the same resolution.
 * For running on the jetson changing the sizes to have a hieght of 464
 * pixels yield around 17 fps. Desktop applications this will run very fast
 *
 * The program works by first selecting the area of the soccer goal you
 * want to track (ie the upper left corner) in all the training photos
 * displayed. Then go select where to find the corner in the first frame
 * of the video. It will then begin tracking automatically and if you hit
 * 'p' it will pause the video and show computation time and fps. Hitting
 * 'd' will toggle debugging mode which will show or hide the frames as it
 * processes them. Turning it off will process frames quicker on the Jetson.
 *
 * The main algorithm we are using here is template matching. It compares a
 * small image to the current frame and checks every spot where and tries to
 * find the best match. This is very slow algorithm. So to speed it up instead
 * of searching the whole frame we are using a kalman filter to predict where
 * we think the match will be in the next frame and search an area that is a
 * 1.5 times greater in each dimension that the image we are searching for. If
 * prediction were accurate then the area can be small, if not the area has to
 * be larger. Getting data from the IMU and extending the kalmand filter would
 * be a good way to improve prediction. However in this program we don't have
 * that data since the video was taken with a cell phone camera. The measurement
 * error of the match is near 0 so we can assume use a very low value for the
 * measurement error in the kalman filter. The images we are comparing are gray
 * scale and is currently setup to track the white goal. To do an object in a
 * different color changes will have to be made to the filtering and image
 * processing.
 *
 * This is not currently able to run real time on the Jetson at a
 * resolution of about 830x460, but it proved to be the most reliable way to
 * track the corner of the goal. One way to get faster times on larger images
 * is to use a image pyramid and might be worth time experimenting with if one
 * want to track at 720 or greater. Also a custom template match could be written
 * to instead of checking every pixel increment since there should be enough overlap
 * in pixels that an accurate enough placement could be found by shifting the image
 * over by 2 pixels at each check instead of 1 pixel.
 *
 *
 *  TRAINING: We train on multiple images in case one image doesn't give us a good
 *  enough match as the perspective changes because of camera or object movement.
 *  Images are currently saved and stored in a list. Because we don't want to have
 *  to run template match more than once we update the list when the first image
 *  doesn't meet our threshold to have a great match to be the image that is found
 *  to have a great match. This way the next time we search an image the list will use
 *  the first image in the list (hopefully a great match still) and only check one
 *  image.
 */
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <ctype.h>
#include <vector>
#include <sys/time.h>
#include <stdio.h>

using namespace cv;
using namespace cv::gpu;
using namespace std;

Mat image, frame0, gray, sh;
GpuMat gpu_frame0, gpu_gray, gpu_mask, gpu_thresh, gpu_temp;
vector<GpuMat> train_coll(8), mask_coll(8);
vector<Rect> selections(8);

const int thresh = 200; //threshold value for minimum gray scale value
int element_shape = MORPH_RECT; //use a rectangle for erode and dialte

Rect selection; //rectangle used for masking and selecting the area we want to track
Point origin; //used in selecting
bool selectObject = false; //state variable for selecting the object
int trackObject = 0; //state variable to indicate if we should be tracking the object

/*
 * The function selects the object in the window
 */
static void onMouse( int event, int x, int y, int, void* )
{
	if( selectObject )
	{
		//get top left corner
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		//set dimensions of selection
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);

		selection &= Rect(0, 0, image.cols, image.rows);
	}

	switch( event )
	{
	case CV_EVENT_LBUTTONDOWN: //started selecting
		origin = Point(x,y);
		selection = Rect(x,y,0,0);
		selectObject = true;
		break;
	case CV_EVENT_LBUTTONUP:
		selectObject = false;
		if( selection.width > 0 && selection.height > 0 ) //done selecting and selection is valid
			trackObject = -1; //indicate we should track object
		break;
	}
}
/*
 * used to calculate the difference in time measurements
 */
long getTimeDelta(struct timeval timea, struct timeval timeb) {
	return 1000000 * (timeb.tv_sec - timea.tv_sec) +
			(int(timeb.tv_usec) - int(timea.tv_usec));
}
/*
 * kalman_init
 * initializes the kalman filter to a starting state.
 * Inputs:
 * 		kal -> reference of a kalman filter object that is not null
 * 		p  	-> starting point of the object
 * 		processNoiseCov -> value of the noise that can happen during processing
 * 		measurementNoiseCov -> value representing noise of the measurement. Make this value smaller if the cv algorithm can accurately identify the location
 * 		errorCovPost -> value of the error somewhere kalman filter so tuning is required
 */
void kalman_init(KalmanFilter &kal, Point p, double processNoiseCov, double measurementNoiseCov, double errorCovPost){
	kal.statePre.at<float>(0) = p.x;
	kal.statePre.at<float>(1) = p.y;
	kal.statePre.at<float>(2) = 0;
	kal.statePre.at<float>(3) = 0;
	kal.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);

	setIdentity(kal.measurementMatrix);
	setIdentity(kal.processNoiseCov, Scalar::all(processNoiseCov));
	setIdentity(kal.measurementNoiseCov, Scalar::all(measurementNoiseCov));
	setIdentity(kal.errorCovPost, Scalar::all(errorCovPost));
}
/*
 * box_update
 * Updates the box that we are going to search for the template
 * match in to be centered over the center of the current match
 * location.
 *
 * Inputs:
 * 		kal -> reference of a kalman filter object that is used to track and predict the match location
 * 		bb 	-> rectangle that is the size of the training image used to find the best point and is in the location of the match
 *
 * Outputs:
 * 		measurement -> the point to store the measured point
 * 		ctr_point -> the center point of the best match.
 * 		kal_point -> the center point adjusted by the kalman filter
 */
void box_update(KalmanFilter &kal, Rect &bb, Mat_<float> &measurement, Point2f &ctr_point, Point2f &kal_point){
	Point measp = Point(bb.tl().x + (bb.width / 2), bb.tl().y + (bb.height / 2));
	//save measured matrix
	measurement(0) = measp.x;
	measurement(1) = measp.y;

	ctr_point = measp; //save measured center point

	Mat estimated = kal.correct(measurement); //correct the predicted center with the measurement point
	Point statePt(estimated.at<float>(0),estimated.at<float>(1)); //the corrected center
	kal_point = statePt; //save the corrected point
	//update selection so next time we have the mask in the right spot
	selection.x = statePt.x - (bb.width/2);
	selection.y = statePt.y - (bb.height/2);
	selection.width = bb.width;
	selection.height = bb.height;
}
/*
 * process_frame
 * This function does the process to the current frame in the feed
 * that is used to setup the image for template matching
 *
 * frame to process: gpu_frame0
 * output of process to: gpu_gray
 * frames overwritten: gpu_gray, gpu_thresh
 *
 * Inputs:
 * 		morphElement -> the element used for erosion and dilation
 * 		threshold -> the minimal gray scale value to include in the image
 */
void proccess_frame(Mat morphElement, int threshold){
	gpu::cvtColor(gpu_frame0, gpu_gray, COLOR_BGR2GRAY); //convert to gray scale
	//gpu_mask = GpuMat(gpu_gray.size(), CV_8UC1, Scalar::all(0));
	//gpu_mask(selection).setTo(Scalar::all(255));

	gpu::threshold(gpu_gray, gpu_thresh, threshold, 255, THRESH_BINARY); //get rid of pixels less than the threshold
	gpu::bitwise_and(gpu_gray, gpu_thresh, gpu_gray); //get the image with just the pixels that have values >= threshold
	morphologyEx(gpu_gray, gpu_gray, CV_MOP_OPEN, morphElement); //perform erosion and dilation

}
/*
 * match_template
 * runs template match on algorithm and saves the best location and which training
 * image that it was found with.
 *
 * Inputs:
 * 		test  	 <- current frame in video feed to find match in. should already be processed
 * 		train 	 <- list of images to be used to find a match. These should already be processed
 * 		index 	 <- list of indexes associated to the image in train. The first list in the index should be the best match
 * 		best_val <- should be 0 to start with.
 * Outputs:
 * 		best_val <- this is updated to the found match value
 * 		best_loc <- this is set to the location best_val was retrieved from
 * 		index 	 <- this is updated to reflect any change if the best value wasn't obtained from the image at index(0)
 * 		idx		 <- the iteration of train images that used to find the best match
 */
void match_template(GpuMat &test, vector<GpuMat> &train, vector<int> &index, double &best_val, Point &best_loc, int &idx){
	int itr = 0;

	for(int i = 0; i < train_coll.size(); i++){
		itr++;
		gpu::matchTemplate(test, train[index[i]], gpu_temp, CV_TM_CCORR_NORMED); //match template function, using a normalized correlation value as the score
		double max_value;
		Point location;
		gpu::minMaxLoc(gpu_temp, 0, &max_value, 0, &location); //get the best value and location
		if(max_value > best_val){ //update if best value
			best_loc = location;
			best_val = max_value;
			idx = i;
		}
		if (max_value > .80){ //if value scores greater than .80 then it is a good enough match and we can stop. Otherwise try another image
			break;
		}
	}
	if(idx != 0){	//if the image that was used to find the best match isn't the best one then move the one that was to the front of the list.
		int tmp = index.at(idx);
		for(int i = idx; i > 0; i--){
			index[i] = index[i-1];
		}
		index[0] = tmp;
	}
	cerr << itr << " iterations " << best_val << endl;
}

int main( int argc, const char** argv )
{

	VideoCapture cap;
	Rect trackWindow;

	struct timeval timea, timeb, timeS, timeE;
	long totalTime = 0, matchTime = 0, convertTime = 0, loadTime = 0;
	int nFrames = 0;

	cap.open("/home/ubuntu/Aerial/photos/SoccerGoal2_464.mp4"); //open smaller video file (reccomended for Jetson)
//	cap.open("/home/scott/Aerial//aerial_navigation/photos/SoccerGoal2.mp4"); //open regular video file (desktop)

	cerr << cap.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
	cerr << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
	vector<string> screenshots;
	//smaller training images (Jetson)
	screenshots.push_back("/home/ubuntu/Aerial/WicketTraining/sh1_464.png");
	screenshots.push_back("/home/ubuntu/Aerial/WicketTraining/sh2_464.png");
	screenshots.push_back("/home/ubuntu/Aerial/WicketTraining/sh3_464.png");
	screenshots.push_back("/home/ubuntu/Aerial/WicketTraining/sh4_464.png");
	screenshots.push_back("/home/ubuntu/Aerial/WicketTraining/sh5_464.png");
	screenshots.push_back("/home/ubuntu/Aerial/WicketTraining/sh6_464.png");
	screenshots.push_back("/home/ubuntu/Aerial/WicketTraining/sh7_464.png");
	screenshots.push_back("/home/ubuntu/Aerial/WicketTraining/sh8_464.png");
	//regular training images (Jetson)
//	screenshots.push_back("/home/scott/Aerial/aerial_navigation/WicketTraining/sh1.png");
//	screenshots.push_back("/home/scott/Aerial/aerial_navigation/WicketTraining/sh2.png");
//	screenshots.push_back("/home/scott/Aerial/aerial_navigation/WicketTraining/sh3.png");
//	screenshots.push_back("/home/scott/Aerial/aerial_navigation/WicketTraining/sh4.png");
//	screenshots.push_back("/home/scott/Aerial/aerial_navigation/WicketTraining/sh5.png");
//	screenshots.push_back("/home/scott/Aerial/aerial_navigation/WicketTraining/sh6.png");
//	screenshots.push_back("/home/scott/Aerial/aerial_navigation/WicketTraining/sh7.png");
//	screenshots.push_back("/home/scott/Aerial/aerial_navigation/WicketTraining/sh8.png");


	if( !cap.isOpened() ) //make sure video file could be opened
	{
		cout << "***Could not initialize capturing...***\n";
		return -1;
	}
	//define the shape that is to be used for errode and dilate. Change size to increase or decrease the amount erroded and dilated
	Mat element = getStructuringElement(element_shape, Size(3, 3), Point(-1, -1) );

	//Initialize kalman filter
	KalmanFilter KF(4, 2, 0);
	Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));
	Point pt(0, 0);

	//intialize display window
	namedWindow( "TrackingWicket", 0 );
	setMouseCallback( "TrackingWicket", onMouse, 0 );

	Rect bb; //rectangle for used for masking image to decrease template match search time

	//state variables
	bool paused = false;
	bool debug = true;

	cap >> frame0; //load the first frame
	paused = true; //paused for training
	vector<int> index(8); //indexes of the training images
	Point2f ctr_point, kal_point;

	ctr_point = pt; //point for the measured center of matched image
	kal_point = pt; //point for the corrected kalman filter eastimate

	//Gather training images
	for(int i = 0; i < screenshots.size(); i++){
		sh = imread(screenshots[i]); //read in trained image file

		for(;;){

			gpu_frame0.upload(sh); //upload image to gpu memory
			proccess_frame(element, thresh); //process the frame prior to selection

			gpu_gray.download(image); //download processed image so it can be displayed
			if(trackObject < 0) { //part of image has been selected so get the trained image

				mask_coll[i] = GpuMat(gpu_gray.size(), CV_8UC1, Scalar::all(0)); //intialize a mask
				mask_coll[i](selection).setTo(Scalar::all(255)); //set the mask to be the selected area
				gpu::bitwise_and(gpu_gray, mask_coll[i], train_coll[i]); //set the image to be only the parts in the mask
				train_coll[i] = train_coll[i](selection); //set the trained image to be just the size of the selection. I'm not sure that this process is the best way
				selections[i] = selection; //save the selection value for later use
				index[i] = i; // save the index value
				trackObject = 0; //set track object to 0 so we don't repeate this process until we have selected an object
				selectObject = 0; //reset the selection object state to no object
				break;
			}
			if( selectObject && selection.width > 0 && selection.height > 0 ) //if selecting an object show the area being selected
			{
				Mat mask(image, selection);
				bitwise_not(mask, mask);
			}
			imshow("TrackingWicket", image); //display the image
			waitKey(10);
		}
		break;
	}
	//loop over frames in video feed (breaks at end of file)
	for(;;)
	{
		gettimeofday(&timea, NULL); //start overal timer
		if( !paused )
		{
			gettimeofday(&timeS, NULL); //start image load timer
			cap >> frame0; //load next frame
			gettimeofday(&timeE, NULL); //end image load timer
			loadTime += getTimeDelta(timeS, timeE); //add to the load time
			nFrames++; //increment the frames proccessed count
			if( frame0.empty() ) //make sure we have a frame stored
				break;
		}
		if( !paused ) //skip if paused
		{
			if(trackObject < 0) { //if this is first pass through tracking do some initialization
				//set point p to be the center of the selected area
				Point p = Point(selection.tl().x + (selection.width / 2), selection.tl().y + (selection.height / 2));
				bb = selection; //bounding box for the search area is the selection

				kalman_init(KF, p, 1e-4, 1e-4, .1); //initialize kalman filter.

				ctr_point = pt;
				kal_point = pt;

				trackObject = 1; //set this so we don't come through here again
			}
			if(trackObject){

				Mat prediction = KF.predict(); //predict where the center of the match will be
				Point predictPt(prediction.at<float>(0),prediction.at<float>(1)); //get the point
				bool smallwindow = false;
				if(predictPt.x != 0 || predictPt.y != 0){ //if the predicted point isn't the first (which is bad) then use the predicted point to set the search box location
					smallwindow = true;
					selection.x = predictPt.x - (selection.width/2);
					selection.y = predictPt.y - (selection.height/2);
				}

				gettimeofday(&timeS, NULL); //start convert timer
				gpu_frame0.upload(frame0); //upload frame to gpu memory
				proccess_frame(element, thresh); //process the frame

				gettimeofday(&timeE, NULL); //stop convert timer
				convertTime += getTimeDelta(timeS, timeE);
				//gpu_gray.download(gray);

				double best_max_value = 0;
				Point best_location;
				Rect predictRect;
				int idx = 0;
				gettimeofday(&timeS, NULL); //start match template timer
				if(smallwindow){ //if we are using a small window to search for the template

					//change the dimensions of the search box to be 3 times bigger than the size of the train image/selection
					int wt = 1.5*selection.width;
					int ht = 1.5*selection.height;
					//set top left and bottom right locations of search area
					predictRect = Rect(predictPt.x - wt, predictPt.y - ht, predictPt.x + wt, predictPt.y + ht);

					GpuMat roi(gpu_gray, predictRect); //get area of image we want to search

					match_template(roi, train_coll, index, best_max_value, best_location, idx); //run template match

				} else //search the whole image (slow)
					match_template(gpu_gray, train_coll, index, best_max_value, best_location, idx); //run template match

				gettimeofday(&timeE, NULL); //end template match timer
				matchTime += getTimeDelta(timeS, timeE);

				if (best_max_value > .8){ //if the value found was better than .8 the update the found location. Otherwise we didn't find a good enough spot (this is not tuned and can be changed)
					if(smallwindow){
						best_location.x = best_location.x + predictRect.tl().x;
						best_location.y = best_location.y + predictRect.tl().y;
						bb = Rect(best_location.x,best_location.y, selections[index[idx]].width, selections[index[idx]].height);//box is now the size of the matched image and the location of the best fit
						box_update(KF, bb, measurement, ctr_point, kal_point); //update the current location of the image and bounding box
					} else {
						bb = Rect(best_location.x,best_location.y, selections[index[idx]].width, selections[index[idx]].height);
						box_update(KF, bb, measurement, ctr_point, kal_point);
					}
				} else //the object wasn't in our window so search the whole image to find it.
					smallwindow = false;

			}
		}


		if( trackObject < 0 ) {
			paused = false;
		}

		gettimeofday(&timeb, NULL); //stop total timer
		totalTime += getTimeDelta(timea, timeb);
		if(debug){ //if debugging then display the image and rectangles of where the kalman filter (red) things the best spot is and where the matched (yellow) spot is
			frame0.copyTo(image);
			rectangle(image, selection, Scalar(0, 0, 255), 1, 8, 0);
			rectangle(image, bb, Scalar(0, 255, 255), 1, 8, 0);
			circle( image, kal_point, 4, Scalar(0, 0, 255), -1, 8, 0 );
			circle( image, ctr_point, 4, Scalar(0, 255, 255), -1, 8, 0 );
			imshow( "TrackingWicket", image );
		}

		char c = (char)waitKey(10);
		if( c == 27 )
			break;
		switch(c)
		{
		case 'c':
			trackObject = 0;
			break;
		case 'd':
			debug = !debug;
			break;
		case 'p':
			paused = !paused;
			cout << "frames                       : " << nFrames << endl;
			cout << "TotalTime                    : " << double(totalTime)/1000000.0 << endl;
			cout << "FPS                          : " << double(nFrames)/(double(totalTime)/1000000.0) << endl;
			cout << "Percentage Convert Time      : " << double(convertTime)/double(totalTime) << endl;
			cout << "Percentage Match Time        : " << double(matchTime)/double(totalTime) << endl;
			cout << "Percentage Load Time         : " << double(loadTime)/double(totalTime) << endl;
			totalTime = matchTime = convertTime = loadTime = 0;
			nFrames = 0;
			break;
		default:
			;
		}
	}

	return 0;
}
