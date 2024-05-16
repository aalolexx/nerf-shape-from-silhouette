from pipeline_util.pipeline import Context
from pipeline_util.pipeline import *
from pipeline_util.utils import *

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# TODO AEH use huggingface transformers here

class ImageBinarizer:
    def __init__(self, images_raw_path: str,
                 images_depth_path: str,
                 images_segmented_path: str,
                 closing_kernel_size: int = 5) -> None:
        self._images_raw_path = images_raw_path
        self._images_depth_path = images_depth_path
        self._images_segmented_path = images_segmented_path
        self._closing_kernel_size = closing_kernel_size

    def __call__(self, context: Context, next_step: NextStep) -> None:
        # Prepare Sam Image Segmentation Model
        sam = sam_model_registry["default"](checkpoint="data/misc/sam_vit_h_4b8939.pth")
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8
        )

        closing_kernel = np.ones((self._closing_kernel_size, self._closing_kernel_size), np.uint8)

        os.makedirs(self._images_segmented_path, exist_ok=True)

        # Loop trough all images
        for filename in os.listdir(self._images_raw_path):
            if is_image(filename):
                input_image_path = os.path.join(self._images_raw_path, filename)
                depth_image_path = os.path.join(self._images_depth_path,  filename)
                output_image_path = os.path.join(self._images_segmented_path, filename)

                input_image = load_and_resize_image(input_image_path, 720)
                depth_image = load_and_resize_image(depth_image_path, 720).convert("L") # make sure mask is grayscale
                #depth_image.show()

                black_image = Image.new('RGB', input_image.size, (0, 0, 0, 255))
                filtered_image = Image.composite(input_image, black_image, depth_image)
                #filtered_image.show()

                # Actual mask creation
                masks = mask_generator.generate(np.asarray(filtered_image))

                plt.figure(figsize=(20, 20))
                plt.imshow(input_image)
                self.show_anns(masks)
                plt.axis('off')
                plt.show()

                # Get the requested mask and create binarized image
                # TODO aeh find a sofisticated way how to get the correct mask (e.g. center point)
                binary_mask = (masks[3]['segmentation']).astype(int)

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

                #break # TODO AEH REMOVE

        # Call the next module
        next_step(context)


    # Visualize Masks
    # SRC: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)