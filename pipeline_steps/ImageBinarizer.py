from pipeline_util.pipeline import Context
from pipeline_util.pipeline import *

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import os
import cv2

class ImageBinarizer:
    def __init__(self, images_raw_path: str,
                 images_segmented_path: str,
                 closing_kernel_size: int = 5) -> None:
        self._images_raw_path = images_raw_path
        self._images_segmented_path = images_segmented_path
        self._closing_kernel_size = closing_kernel_size

    def __call__(self, context: Context, next_step: NextStep) -> None:
        # Prepare Sam Image Segmentation Model
        sam = sam_model_registry["default"](checkpoint="data/misc/sam_vit_h_4b8939.pth")
        mask_generator = SamAutomaticMaskGenerator(sam)

        closing_kernel = np.ones((self._closing_kernel_size, self._closing_kernel_size), np.uint8)

        os.makedirs(self._images_segmented_path, exist_ok=True)

        # Loop trough all images
        for filename in os.listdir(self._images_raw_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                input_image_path = os.path.join(self._images_raw_path, filename)
                output_image_path = os.path.join(self._images_segmented_path, f"mask_{filename}")

                # todo aeh resize image

                # Load Image
                input_image = Image.open(input_image_path)
                if input_image.mode != 'RGB':
                    input_image = input_image.convert('RGB')

                mask = mask_generator.generate(np.asarray(input_image))
                binary_mask = (mask[0]['segmentation']).astype(int)

                # Convert to binary representation (foreground as white, background as black)
                binary_mask[binary_mask == 1] = 255
                binary_mask = binary_mask.astype(np.uint8)

                # Close small error holes
                binary_mask = cv2.dilate(binary_mask, closing_kernel, iterations=1)
                binary_mask = cv2.erode(binary_mask, closing_kernel, iterations=1)

                binary_image = Image.fromarray(binary_mask.astype('uint8'), 'L')

                # Save the mask image
                binary_image.save(output_image_path)
                print('saved image mask: ' + output_image_path)

        # Call the next module
        next_step(context)
