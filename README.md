# SnowReducer-ScienceFair2024
This project is developing a program to minimize the amount of snow in a video so that autonomous vehicles can detect road signs and lane markers to improve their navigation in inclement weather.


# Dependencies
-Moviepy                 
-Ultralytics                    
-OpenCV-Python                 
-Numpy                           
-Pillow
-Shutil

Run this command in your command prompt to install dependencies:                                         
```pip install moviepy ultralytics opencv-python numpy pillow shutil``` 



# Example


Input Video:


https://github.com/nityashah1811/SnowReducer-ScienceFair2024/assets/102633096/00b80e5e-a0c0-4960-8cba-d4399ded4cbd

Output Video:


https://github.com/nityashah1811/SnowReducer-ScienceFair2024/assets/102633096/3ec09419-8749-4439-ba7f-aeb9a136f941





While not all of the snow is removed all of the big ones which cause real obstructions are removed. The small ones do not interfere too much with detection therefore it is okay if they are not completely removed. There is also an outline remainging and this is also due to lack of training, to improve this I plan on training another model to detect this outline and run this as another pass to maximize the amount of snow removed. There are some black holes in the video and this is due to lack of training as well. 

Note: With more training and annotating this model will become much more accurate to detect snow, with more training even smaller snowflakes can be removed accurately.


# Example Case In Real Life

While the car is driving it will constantly run this program every 0.1 seconds on the live stream from the camera in the car. Then it will remove the snow and output an image, in 1 second there will be about 10 images. if we put those together in a video we can have a 1 second video with no snow. Keep doing this while the car is running then detecting objects such as lane markers and road signs will become so much easier.
