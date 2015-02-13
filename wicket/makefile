default: all

all: canny hough main

canny: cannyEdge.cpp
	g++ -c cannyEdge.cpp

hough: houghLine.cpp
	g++ -c houghLine.cpp

main: main.cpp
	g++ main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_contrib -lopencv_gpu -lopencv_stitching -lopencv_video -lopencv_videostab houghLine.o cannyEdge.o -o prog

#main: main.cpp
#	g++ main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab cannyEdge.o houghLine.o -o prog

testCanny: cannyEdgeTest.cpp
	g++ cannyEdgeTest.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab -o prog

testHough: houghLineTest.cpp
	g++ houghLineTest.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab -o prog


clean:
	rm prog cannyEdge.o houghLine.o
