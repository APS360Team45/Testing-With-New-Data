import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.open(r"C:\Users\MSadm\Downloads\photos\Banana\0\20230803_192104.jpg")

def convert_images(folder):
    # Rename all files to .jpg
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".heic"):
                input_path = os.path.join(root, file)
                output_path = os.path.splitext(input_path)[0] + ".jpg"

                # Rename the file to .jpg
                os.rename(input_path, output_path)

    # Resize all .jpg files to 480p
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                input_path = os.path.join(root, file)
                print(input_path)
                
                # Open the JPEG image with PIL and downscale it to 480p
                with Image.open(input_path) as img:
                    aspect_ratio = img.width / img.height

                    if aspect_ratio < 1:
                        new_width = 854
                        new_height = int(new_width / aspect_ratio)
                    else:
                        new_height = 480
                        new_width = int(new_height * aspect_ratio)

                    img = img.resize((new_width, new_height))

                    # Save the downscaled image
                    img.save(input_path)

FOLDER = r"photos"
convert_images(FOLDER)