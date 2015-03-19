# aerial_navigation
Aerial navigation software

This is software and files used for all things navigation
Some of the files in here are generated from Nvidia Nsight Ecllipse for use with development on the Jetson

Programs: 
  BalloonTracker - Implements Camshift algoirthm meant to track a red balloon
  IdentifyBalloon_Camera - Identifies red balloon in video feed from a camera based on roundness of red objects
  TestImage_Detect - This was used to test the identification of balloons at various distances
  TrackingFilter_Tunner - Has the ability to test out different filter setting in a camera feed
  WicketTracking - Implements a fast template match with a kalman filter. Tracks a soccer goal as a test for a wicket.
  wicket - Meant to identify wicket. Photos and files needed for doing line analysis on the wicket included.
  
Data:
  WicketTraining - training photos required for WicketTracking
  outPhotos - photos of the results from TestImage_Detect
  photos - photos of balloon at different distances, videos of balloon, and videos of soccer goal (required for WicketTracking)
  

Contact Scott or Matt if you have questions
