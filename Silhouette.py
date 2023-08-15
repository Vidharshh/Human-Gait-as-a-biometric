import os
from PIL import Image, ImageOps

root_dir = "D:/College/6th Sem/CIP/Coding/frames_output"
output_dir = "D:/College/6th Sem/CIP/Coding/frames_silhouette"

# Loop through all the files in the input directory
for filename in os.listdir(root_dir):
    # Check if the file is an image file
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open the image file
        img_path = os.path.join(root_dir, filename)
        img = Image.open(img_path)

        # Convert the image to grayscale
        img_gray = img.convert('L')

        # Invert the grayscale image
        img_gray_inv = ImageOps.invert(img_gray)

        # Threshold the inverted grayscale image to create the silhouette
        threshold_value = 80
        img_silhouette = img_gray_inv.point(lambda x: 255 * (x > threshold_value))

        # Save the output image
        output_path = os.path.join(output_dir, filename.split(".")[0] + "_s.png")
        img_silhouette.save(output_path)
