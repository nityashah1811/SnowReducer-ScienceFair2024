# IMPORT DEPENDENCIES
from moviepy.editor import VideoFileClip
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import shutil


# SPLIT LONG VIDEO INTO SHORTER SEGMENTS
def split_video(video_file, output_dir, segment_length=0.2, target_fps=60):
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    clip = VideoFileClip(video_file)
    duration = clip.duration
    segments = int(duration / segment_length)
    # Go through the duration of the video and save 0.2 second segments of the video.
    for i in range(segments):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        segment = clip.subclip(start_time, end_time)
        output_path = os.path.join(output_dir, f"segment_{i}.mp4")
        segment.write_videofile(output_path, fps=target_fps)

    clip.close()


# SNOW REMOVAL FUNCTION


# VIDEO SPLITTER INTO FRAMES
def extract_frames(movie, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    clip = VideoFileClip(movie)
    times = (i / clip.fps for i in range(int(clip.fps * clip.duration)))
    # Iterate through the short videos and save all frames of the video
    for t in times:
        imgpath = os.path.join(output_folder, "{}.png".format(int(t * clip.fps)))
        clip.save_frame(imgpath, t)


# PREDICTING SNOW IN THE IMAGE AND OUTPUTTING MASK OF PREDECTIONS
# FIRST PASS
def generate_masks(mask_folder, model_path, input_folder, condfidence_interval):

    # Create the output mask folder
    if os.path.exists(mask_folder):
        shutil.rmtree(mask_folder)
    os.makedirs(mask_folder)

    # Load the YOLO model
    model = YOLO(model_path)

    # Iterate through images in the input folder
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        output_filename = os.path.join(mask_folder, f'mask_{filename}')

        img = cv2.imread(image_path)
        H, W, _ = img.shape

        # Perform inference
        results = model(img, conf=condfidence_interval)

        # Initialize an empty mask to accumulate detections
        combined_mask = np.zeros((H, W), dtype=np.uint8)

        for result in results:
            for j, mask in enumerate(result.masks.data):
                mask = mask.numpy()
                mask = cv2.resize(mask, (W, H))

                # Accumulate the masks
                combined_mask = np.maximum(combined_mask, mask)

        # Save the combined mask
        cv2.imwrite(output_filename, combined_mask * 255)



# REMOVING THE PREDICTED SNOW FROM THE IMAGE
def remove_predicted_snow(input_folder, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Iterate through images in the input folder
    for filename in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, filename)
        mask_path = os.path.join(mask_folder, f'mask_{filename}')
        output_image_path = os.path.join(output_folder, f'output_{filename}')

        # Open the input image and mask
        input_image = Image.open(input_image_path).convert('RGBA')
        mask = Image.open(mask_path).convert('L')

        # Create a new image with the same size as the input image
        output_image = Image.new('RGBA', input_image.size)

        # Get the pixel data from the images
        input_pixels = input_image.load()
        mask_pixels = mask.load()
        output_pixels = output_image.load()

        # Process every pixel
        for y in range(input_image.height):
            for x in range(input_image.width):
                # If the mask pixel is white, make the output pixel transparent
                if mask_pixels[x, y] == 255:
                    output_pixels[x, y] = (0, 0, 0, 0)
                else:
                    output_pixels[x, y] = input_pixels[x, y]

        # Save the output image
        output_image.save(output_image_path)



# SECOND PASS TO MAXIMIZE AMOUNT OF SNOW REMOVED


# Overlay images
def overlay_images(output_filename, removed_pngs_folder):
    if os.path.exists("FinalOutputPngs2nd"):
        shutil.rmtree("FinalOutputPngs2nd")
    os.makedirs("FinalOutputPngs2nd", exist_ok=True)



    # Get a list of all image files in the input folder
    image_files = [file for file in os.listdir(removed_pngs_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

    # Ensure there are images in the folder
    if not image_files:
        print("No images found in the folder.")
    else:
        # Load the first image
        result = Image.open(os.path.join(removed_pngs_folder, image_files[0])).convert("RGBA")

        # Iterate over the remaining images and composite them onto the result
        for file in image_files[1:]:
            current_image = Image.open(os.path.join(removed_pngs_folder, file)).convert("RGBA")
            result = Image.alpha_composite(result, current_image)

        # Save the final result
        result.save(output_filename)




# ORGANIZE IMAGES NUMERICALLY TO ENSURE THE VIDEO IS CREATED IN THE RIGHT ORDER
def organize_images(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files in input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]

    # Sort image files numerically
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Copy sorted images to output folder
    for idx, image_file in enumerate(image_files, start=10000):
        shutil.copyfile(os.path.join(input_folder, image_file), os.path.join(output_folder, f'{idx}.png'))


# CONVERT SNOW REMOVED IMAGES TO VIDEO
def images_to_video(image_folder, video_name, fps, frame_duration):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        for _ in range(frame_duration):  # Repeat frame for frame_duration times
            video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


# CALL THE FUNCTIONS

split_video("20sVideoTest.mp4", "ShorterVideos")

shorter_videos_folder = "ShorterVideos"


# ITERATE THROUGH EACH VIDEO RUNNING THE SNOW_REMOVED FUNCTION ON EACH ONE
for filename in os.listdir(shorter_videos_folder):
    if filename.endswith('.mp4'):
        video_path = os.path.join(shorter_videos_folder, filename)
        movie = str(video_path)
        unremoved_pngs_folder = str(filename) + "videoPngs"

        extract_frames(movie, unremoved_pngs_folder)
        mask_folder = str(filename) + 'videoMasks'
        generate_masks(mask_folder, "last.pt", unremoved_pngs_folder, 0.4)
        output_folder = str(filename) + 'videoOutputPngs'

        remove_predicted_snow(unremoved_pngs_folder, output_folder)
        mask_folder = str(filename) + 'videoMasks2nd'
        generate_masks(mask_folder, "SnowExcess.pt", output_folder, 0.1)
        output_folder_2nd = str(filename) + "videoOutputPngs2nd"
        remove_predicted_snow(output_folder, output_folder_2nd)

        output_filename = "FinalOutputPngs2nd/" + str(filename) + "videoFinal2nd.png"
        removed_pngs_folder = str(filename) + 'videoOutputPngs2nd'
        overlay_images(output_filename, removed_pngs_folder)

image_folder = "FinalOutputPngs2nd"
video_name = "output.mp4"
fps = 60  # Frames per second
frame_duration = 12  # Duration to display each image (in frames)  to convert to seconds do seconds for frame times fps0
organized_images = "OraganizedPngs"

organize_images(image_folder, organized_images)

images_to_video(organized_images, video_name, fps, frame_duration)
