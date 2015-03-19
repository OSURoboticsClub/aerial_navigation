You can either run the program on an image/video within the same directory, or run from a camera feed.

to use image file as source:
   make
   ./prog <image.JPG>

to use video file as source:  **not implemented yet
   make
   ./prog <videofile>

to use camera feed as source:
   **make sure camera/webcam is plugged in to Jetson**
   make
   ./prog


To apply different filters:
   go into main.cpp and comment out or uncomment defines in the "control center"
   note that filters are applied in the order they are defined
   use different combinations of filters to achieve different results
