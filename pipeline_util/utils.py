from PIL import Image
import os

def load_and_resize_image(image_path, new_image_width):
    # Load Image
    input_image = Image.open(image_path)
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')

    new_width = new_image_width
    new_height = int(input_image.height * (new_width / input_image.width))

    return input_image.resize((new_width, new_height))

def is_image(filename):
    return filename.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))