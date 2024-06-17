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

class ImageBinarizerRMBGNoDepth:
    def __init__(self, images_raw_path: str,
                 images_segmented_path: str,
                 target_width: int = -1) -> None:
        self._images_raw_path = images_raw_path
        self._images_segmented_path = images_segmented_path
        self._target_width = target_width

    def __call__(self, context: Context, next_step: NextStep) -> None:

        os.makedirs(self._images_segmented_path, exist_ok=True)

        # Loop trough all images
        for filename in os.listdir(self._images_raw_path):
            if is_image(filename):
                input_image_path = os.path.join(self._images_raw_path, filename)
                output_image_path = os.path.join(self._images_segmented_path, filename)

                input_image_downscaled = load_and_resize_image(input_image_path, self._target_width)

                # Actual mask creation
                pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
                pillow_mask = pipe(input_image_downscaled, return_mask=True)  # outputs a pillow mask

                # CLosing and Binarizing
                pillow_mask = pillow_mask.filter(ImageFilter.MaxFilter(9))
                pillow_mask = pillow_mask.filter(ImageFilter.MinFilter(9))
                pillow_mask = pillow_mask.point(lambda p: 255 if p > 150 else 0)

                # Save the mask image
                pillow_mask.save(output_image_path)
                print('saved image mask: ' + output_image_path)

        # Call the next module
        next_step(context)
