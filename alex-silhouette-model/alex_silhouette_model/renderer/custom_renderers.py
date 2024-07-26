import contextlib
import math
from typing import Generator, Literal, Optional, Tuple, Union

import nerfacc
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.utils import colors
from nerfstudio.utils.math import components_from_spherical_harmonics, safe_normalize

BackgroundColor = Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"], Float[Tensor, "*bs 3"]]
BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None


class BWRenderer(nn.Module):
    """Standard volumetric rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    def __init__(self,
                 background_color: BackgroundColor = "random",
                 use_optimized_sigmoid: bool = True,
                 use_weight_prioritization: bool = True,
                 sig_range: float = 8.4,
                 sig_offset: float = 4.2
                 ) -> None:

        super().__init__()

        self.background_color: BackgroundColor = background_color
        self.use_optimized_sigmoid = use_optimized_sigmoid
        self.use_weight_prioritization = use_weight_prioritization
        self.sig_range = sig_range
        self.sig_offset = sig_offset

    def improved_sigmoid(self, x):
        # return 1/(1+math.exp(-x * c + a))
        c = self.sig_range
        a = self.sig_offset
        return 1 / (1 + torch.exp((-1 * x) * c + a))


    def weight_priority_function(self, x):
        return -1 * torch.pow(x * 1.4 - 0.6, 2) + 1.2

    def combine_rgb(
        self,
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_samples: RaySamples,
    ) -> Float[Tensor, "*bs 1"]:
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

        # CHANGE
        # comp_rgb = torch.sum(weights * rgb, dim=-2)
        # rgb.shape -> (1024, 48, 3) = (pixels/rays, samples, rgb)
        # weights.shape -> (1024, 48, 1) = (pixels/rays, samples, weight value)
        # comp_rgb.shape -> (1024, 3) = (pixels/rays, rgb)

        # Create an index tensor of the weights, for later prioritizing of the weights
        pixel_dim, sample_dim, value_dim = weights.shape
        weight_index = torch.arange(sample_dim).view(1, sample_dim, value_dim).expand(pixel_dim, sample_dim, value_dim)
        weight_index = weight_index / sample_dim  # normalize
        weight_priorities = self.weight_priority_function(weight_index).to(weights.device)

        # Apply Weight Priorization and sum along ray to get color value (Value between 0 and 1)
        if self.use_weight_prioritization:
            accumulated_weight = torch.sum(torch.multiply(weights, weight_priorities), dim=-2)  # -2 = samples per ray (dimension)
        else:
            accumulated_weight = torch.sum(weights, dim=-2)  # -2 = samples per ray (dimension)

        if self.use_optimized_sigmoid:
            accumulated_weight = self.improved_sigmoid(accumulated_weight)

        # Final Binarization
        #accumulated_weight[accumulated_weight >= 0.5] = 1
        #accumulated_weight[accumulated_weight < 0.5] = 0
        return accumulated_weight  # Shape 4096, 1

    def get_background_color(
        self, background_color: BackgroundColor, shape: Tuple[int, ...], device: torch.device
    ) -> Union[Float[Tensor, "1"], Float[Tensor, "*bs 1"]]:
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

        # CHANGE
        # Ensure correct shape
        return torch.tensor([0.0]).to(device)

    def blend_background(
        self,
        image: Tensor,
        background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs 1"]:
        """Blends the background color into the image if image is RGBA.
        Otherwise no blending is performed (we assume opacity of 1).

        Args:
            image: RGB/RGBA per pixel.
            opacity: Alpha opacity per pixel.
            background_color: Background color.

        Returns:
            Blended RGB.
        """
        if image.size(-1) < 4:
            return image

        rgb, opacity = image[..., :1], image[..., 1:]
        background_color = self.get_background_color(self.background_color, shape=rgb.shape, device=rgb.device)
        assert isinstance(background_color, torch.Tensor)
        return rgb * 1 + background_color.to(rgb.device) * (1 - opacity)

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
        gt_image = self.blend_background(gt_image, background_color=self.background_color)
        return pred_image, gt_image

    def forward(
        self,
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_samples: RaySamples,
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 1"]:
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

        rgb = self.combine_rgb(weights, ray_samples)

        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)

        return rgb