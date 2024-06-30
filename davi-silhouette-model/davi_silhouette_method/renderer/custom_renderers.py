
from torch import Tensor, nn



import contextlib
from typing import Generator, Literal, Optional, Tuple, Union

import nerfacc
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from nerfstudio.utils import colors

BackgroundColor = Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"], Float[Tensor, "*bs 3"]]
# BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None


# @contextlib.contextmanager
# def background_color_override_context(mode: Float[Tensor, "3"]) -> Generator[None, None, None]:
#     """Context manager for setting background mode."""
#     global BACKGROUND_COLOR_OVERRIDE
#     old_background_color = BACKGROUND_COLOR_OVERRIDE
#     try:
#         BACKGROUND_COLOR_OVERRIDE = mode
#         yield
#     finally:
#         BACKGROUND_COLOR_OVERRIDE = old_background_color


class BinaryRenderer(nn.Module):
    """Binary black/white volumetric rendering.
    
    Should replace RGBRenderer().
    """

    def __init__(self) -> None: #, background_color: BackgroundColor = "random") -> None:
        super().__init__()
        self.background_color = torch.tensor([0.0, 0.0, 0.0])

    # TODO: change input to density
    @classmethod
    def combine_rgb(
        cls,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        # ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        # num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image.
        If background color is random, no BG color is added - as if the background was black!

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            background_color: Background color as RGB.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """
        ## Ignore these cases leading to batched samples instead of packed samples

        # if ray_indices is not None and num_rays is not None:
        #     comp_rgb = nerfacc.accumulate_along_rays(
        #         weights[..., 0], values=rgb, ray_indices=ray_indices, n_rays=num_rays
        #     )
        #     accumulated_weight = nerfacc.accumulate_along_rays(
        #         weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
        #     )
        # else:
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        # accumulated_weight = torch.sum(weights, dim=-2)

        # background_color = cls.get_background_color(background_color, shape=comp_rgb.shape, device=comp_rgb.device)

        # assert isinstance(background_color, torch.Tensor)
        # comp_rgb = comp_rgb + background_color * (1.0 - accumulated_weight)
        return comp_rgb
    
    # NOTE: new method for binary - supposed to replace combine_rgb
    @classmethod
    def binary_density(
        cls,
        weights: Float[Tensor, "*bs num_samples 1"],
    ) -> Float[Tensor, "*bs 1"]:
        """Accumulates densities along ray and returns 1 or 0 / white or black in some format

        Args:
            weights: Weights for each sample

        Returns:
            Outputs binary values and/or black/white color.
        """
        accumulated_weight = torch.sum(weights, dim=-2)
        
        # print("-----Accumulated Weight Information-----")
        # print(f"Shape: {accumulated_weight.shape}, Data Type: {accumulated_weight.dtype}, Device: {accumulated_weight.device}")
        # print(f"Min value: {accumulated_weight.min().item()}, Max value: {accumulated_weight.max().item()}")

        # # Print the first few elements of the tensor to inspect the data
        # print("First few elements of accumulated_weight:")
        # print(accumulated_weight[:5])  # Adjust the slicing as needed to get a reasonable view
        
        # Apply thresholding to make values binary
        threshold = 0.5
        binary_accumulated_weight = (accumulated_weight >= threshold).float()

        # # Print detailed information about binary_accumulated_weight
        # print("-----Binary Accumulated Weight Information-----")
        # print(f"Shape: {binary_accumulated_weight.shape}, Data Type: {binary_accumulated_weight.dtype}, Device: {binary_accumulated_weight.device}")
        # print(f"Min value: {binary_accumulated_weight.min().item()}, Max value: {binary_accumulated_weight.max().item()}")

        # # Print the first few elements of the tensor to inspect the data
        # print("First few elements of binary_accumulated_weight:")
        # print(binary_accumulated_weight[:5])  # Adjust the slicing as needed to get a reasonable view

        return binary_accumulated_weight # binary_accumulated_weight


    @classmethod
    def get_background_color(
        cls, background_color: BackgroundColor, shape: Tuple[int, ...], device: torch.device
    ) -> Union[Float[Tensor, "3"], Float[Tensor, "*bs 3"]]:
        """Returns the RGB background color for a specified background color.
        Note:
            This function CANNOT be called for background_color being either "last_sample" or "random".

        Args:
            background_color: The background color specification. If a string is provided, it must be a valid color name.
            shape: Shape of the output tensor.
            device: Device on which to create the tensor.

        Returns:
            Background color as RGB.
        """
        # assert background_color not in {"last_sample", "random"}
        # assert shape[-1] == 3, "Background color must be RGB."
        # if BACKGROUND_COLOR_OVERRIDE is not None:
        #     background_color = BACKGROUND_COLOR_OVERRIDE
        # if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
        #     background_color = colors.COLORS_DICT[background_color]
        # assert isinstance(background_color, Tensor)

        # Ensure correct shape
        return background_color.expand(shape).to(device)

    def blend_background(
        self,
        image: Tensor,
        background_color = None, #: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Blends the background color into the image if image is RGBA.
        Otherwise no blending is performed (we assume opacity of 1).

        Args:
            image: RGB/RGBA per pixel.
            opacity: Alpha opacity per pixel.
            background_color: Background color.

        Returns:
            Blended RGB.
        """
        ## sooner or later it would be not even RGB
        if image.size(-1) < 4:
            return image

        # rgb, opacity = image[..., :3], image[..., 3:]
        # if background_color is None:
        #     background_color = self.background_color
        #     if background_color in {"last_sample", "random"}:
        #         background_color = "black"
        # background_color = self.get_background_color(background_color, shape=rgb.shape, device=rgb.device)
        # assert isinstance(background_color, torch.Tensor)
        # return rgb * opacity + background_color.to(rgb.device) * (1 - opacity)

    def blend_background_for_loss_computation(
        self,
        pred_image: Tensor,
        pred_accumulation: Tensor,
        gt_image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Blends a background color into the ground truth and predicted image for
        loss computation.

        Args:
            gt_image: The ground truth image.
            pred_image: The predicted RGB values (without background blending).
            pred_accumulation: The predicted opacity/ accumulation.
        Returns:
            A tuple of the predicted and ground truth RGB values.
        """
        background_color = self.background_color
        # if background_color == "last_sample":
        #     background_color = "black"  # No background blending for GT
        # elif background_color == "random":
        #     background_color = torch.rand_like(pred_image)
        #     pred_image = pred_image + background_color * (1.0 - pred_accumulation)
        gt_image = self.blend_background(gt_image, background_color=background_color)
        return pred_image, gt_image

    def forward(
        self,
        # rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        # ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        # num_rays: Optional[int] = None,
        # background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.
            background_color: The background color to use for rendering.

        Returns:
            Outputs of rgb values.
        """

        # if background_color is None:
        #     background_color = self.background_color

        # if not self.training:
        #     rgb = torch.nan_to_num(rgb)
        if not self.training:
            weights = torch.nan_to_num(weights)
        
        # print(f"rgb1: Shape: {rgb.shape}, Data Type: {rgb.dtype}, Device: {rgb.device}")
        # print("First rgb1 few elements:", rgb[:3])
        # rgb = self.combine_rgb(
        #     rgb, weights#, background_color=background_color, ray_indices=ray_indices, num_rays=num_rays
        # )
        # print(f"rgb2: Shape: {rgb.shape}, Data Type: {rgb.dtype}, Device: {rgb.device}")
        # print("First rgb2 few elements:", rgb[:3])
        # print(f"Weights: Shape: {weights.shape}, Data Type: {weights.dtype}, Device: {weights.device}")
        # print("First weights few elements:", weights[:3])
        # NOTE: return binary here instead
        binary_output = self.binary_density(weights=weights)
        # print(f"Binary_output: Shape: {binary_output.shape}, Data Type: {binary_output.dtype}, Device: {binary_output.device}")
        # print("First binary_output few elements:", binary_output[:3])
        # if not self.training:
        #     torch.clamp_(rgb, min=0.0, max=1.0)
        if not self.training:
            torch.clamp_(binary_output, min=0.0, max=1.0)
        
        return binary_output