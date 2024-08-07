from pipeline_util.pipeline import Context
from pipeline_util.pipeline import *
from pipeline_util.utils import *

from transformers import pipeline
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# TODO AEH use huggingface transformers here

class ImageBinarizerRMBG:
    def __init__(self, images_raw_path: str,
                 images_depth_path: str,
                 images_segmented_path: str,
                 closing_kernel_size: int = 5,
                 target_width: int = -1,
                 limit: int = -1) -> None:
        self._images_raw_path = images_raw_path
        self._images_depth_path = images_depth_path
        self._images_segmented_path = images_segmented_path
        self._closing_kernel_size = closing_kernel_size
        self._target_width = target_width
        self._limit = limit

    def __call__(self, context: Context, next_step: NextStep) -> None:
        closing_kernel = np.ones((self._closing_kernel_size, self._closing_kernel_size), np.uint8)

        os.makedirs(self._images_segmented_path, exist_ok=True)

        i = 0

        # Loop trough all images
        for filename in os.listdir(self._images_raw_path):
            if self._limit > 0:
                i = i + 1
                if i > self._limit:
                    break

            if is_image(filename):
                input_image_path = os.path.join(self._images_raw_path, filename)
                depth_image_path = os.path.join(self._images_depth_path,  filename)
                output_image_path = os.path.join(self._images_segmented_path, filename)

                input_image = load_and_resize_image(input_image_path, self._target_width)
                depth_image = load_and_resize_image(depth_image_path, self._target_width).convert("L")  # make sure mask is grayscale
                #depth_image.show()

                bg_enhancer = ImageEnhance.Brightness(input_image)
                bg_image = bg_enhancer.enhance(0.2)
                #bg_image.show()

                filtered_image = Image.composite(input_image, bg_image, depth_image)
                #filtered_image.show()

                # Actual mask creation
                pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
                pillow_mask = pipe(filtered_image, return_mask=True)  # outputs a pillow mask

                # Convert to binary representation (foreground as white, background as black)

                pillow_mask = pillow_mask.filter(ImageFilter.MaxFilter(9))
                pillow_mask = pillow_mask.filter(ImageFilter.MinFilter(9))
                pillow_mask = pillow_mask.point(lambda p: 255 if p > 150 else 0)

                # Save the mask image
                pillow_mask.save(output_image_path)
                print('saved image mask: ' + output_image_path)

                #break # TODO AEH REMOVE

        # Call the next module
        next_step(context)
