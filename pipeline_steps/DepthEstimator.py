from pipeline_util.pipeline import Context
from pipeline_util.pipeline import *
from pipeline_util.utils import *

from transformers import pipeline
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


class DepthEstimator:
    def __init__(self, images_raw_path: str,
                 images_segmented_path: str,
                 threshold: int = 110,
                 target_width: int = -1,
                 limit: int = -1) -> None:
        self._images_raw_path = images_raw_path
        self._images_depthmap_path = images_segmented_path
        self._threshold = threshold
        self._target_width = target_width
        self._limit = limit
        print(limit)

    def __call__(self, context: Context, next_step: NextStep) -> None:

        os.makedirs(self._images_depthmap_path, exist_ok=True)

        # TODO Same Loop is in ImageBinarize -> Create util wrapper function for the loop

        depth_estimation_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

        # TODO put in params
        closing_kernel = np.ones((5, 5), np.uint8)

        i = 0

        # Loop trough all images
        for filename in os.listdir(self._images_raw_path):
            if self._limit > 0:
                i = i + 1
                if i > self._limit:
                    break

            if is_image(filename):
                input_image_path = os.path.join(self._images_raw_path, filename)
                output_image_path = os.path.join(self._images_depthmap_path, filename)
                input_image = load_and_resize_image(input_image_path, self._target_width)

                depth = depth_estimation_pipe(input_image)["depth"]

                np_image = np.array(depth)
                np_image[np_image < self._threshold] = 0

                # Close small error holes
                np_image = cv2.dilate(np_image, closing_kernel, iterations=2)
                np_image = cv2.erode(np_image, closing_kernel, iterations=2)

                thresholded_depth = Image.fromarray(np_image)
                thresholded_depth.save(output_image_path)

                print('saved image depth: ' + output_image_path)

                #break # TODO AEH REMOVE

        # Call the next module
        next_step(context)
