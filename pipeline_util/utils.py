from PIL import Image, ImageOps
import os


def load_and_resize_image(image_path, new_image_width=-1):
    # Load Image
    input_image = Image.open(image_path)
    input_image = ImageOps.exif_transpose(input_image)
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')

    if new_image_width != -1:
        new_width = new_image_width
        new_height = int(input_image.height * (new_width / input_image.width))
        return input_image.resize((new_width, new_height))
    else:
        return input_image

def is_image(filename):
    return filename.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))