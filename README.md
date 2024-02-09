# SnowReducer-ScienceFair2024
This project is developing a program to minimize the amount of snow in a video so that autonomous vehicles can detect road signs and lane markers to improve their navigation in inclement weather.



# Example


If an input is a video such as this one (about 0.1 seconds long)

https://github.com/nityashah1811/SnowReducer-ScienceFair2024/assets/102633096/31d95681-6902-4ec5-9d4e-12f24ed0cf60


You will get an output like this

![SnowLarge5Final](https://github.com/nityashah1811/SnowReducer-ScienceFair2024/assets/102633096/c5e83353-37f9-4e0f-bddb-97c148501b89)

If you compare this image to the original video you can see how much snow has been removed. There was a highway sign that was partially covered up and is now clear and visible.

While not all of the snow is removed all of the big ones which cause real obstructions are removed. The small ones do not interfere too much with detection therefore it is okay if they are not completely removed.

Note: With more training and annotating this model will become much more accurate to detect snow, with more training even smaller snowflakes can be removed accurately.


# Example Case In Real Life

While the car is driving it will constantly run this program every 0.1 seconds on the live stream from the camera in the car. Then it will remove the snow and output an image, in 1 second there will be about 10 images. if we put those together in a video we can have a 1 second video with no snow. Keep doing this while the car is running then detecting objects such as lane markers and road signs will become so much easier.
