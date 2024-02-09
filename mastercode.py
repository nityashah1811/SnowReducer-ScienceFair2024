from moviepy.editor import VideoFileClip
import os

def extract_frames(movie, times, imgdir):
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)

    clip = VideoFileClip(movie)
    for t in times:
        imgpath = os.path.join(imgdir, "{}.png".format(int(t * clip.fps)))
        clip.save_frame(imgpath, t)


movie = "video.mp4" #replace with path to video
imgdir = "./videoPngs"
clip = VideoFileClip(movie)
times = (i/clip.fps for i in range(int(clip.fps * clip.duration)))
extract_frames(movie, times, imgdir)




from ultralytics import YOLO
import cv2
import numpy as np
import os

model_path = 'last.pt'
input_folder = 'videoPngs'
output_folder = 'videoMasks'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the YOLO model
model = YOLO(model_path)

# Adjust the confidence threshold here
confidence_threshold = 0.1

# Iterate through images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        output_filename = os.path.join(output_folder, f'mask_{filename}')

        img = cv2.imread(image_path)
        H, W, _ = img.shape

        # Perform inference
        results = model(img, conf = confidence_threshold)

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




from PIL import Image
import os

input_folder = 'videoPngs'
mask_folder = 'videoMasks'
output_folder = 'videoOutputPngs'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
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


from PIL import Image
import os

input_folder = 'videoOutputPngs'
output_filename = 'videoFinal.png'

# Get a list of all image files in the input folder
image_files = [file for file in os.listdir(input_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Ensure there are images in the folder
if not image_files:
    print("No images found in the folder.")
else:
    # Load the first image
    result = Image.open(os.path.join(input_folder, image_files[0])).convert("RGBA")

    # Iterate over the remaining images and composite them onto the result
    for file in image_files[1:]:
        current_image = Image.open(os.path.join(input_folder, file)).convert("RGBA")
        result = Image.alpha_composite(result, current_image)

    # Save the final result
    result.save(output_filename)

    # Show the final result
    result.show()


