"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal, Optional

from torch import Tensor

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field


class TemplateNerfField(NerfactoField):
    """Template Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
    ) -> None:
        super().__init__(aabb=aabb, num_images=num_images)

    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.


# # for subclass CustomHashMLPDensityField
# from nerfstudio.fields.density_fields import HashMLPDensityField
# from nerfstudio.cameras.rays import RaySamples
# from nerfstudio.data.scene_box import SceneBox
# from nerfstudio.field_components.activations import trunc_exp
# from torch import Tensor
# from typing import Tuple

# class CustomHashMLPDensityField(HashMLPDensityField):
#     """Custom density field that binarizes output
#     After calculating density, a threshold of 0.5 is applied and the densities are mapped to either 0 or 1
#     """
#     def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
#         # Apply spatial distortion if specified, otherwise normalize positions within the bounding box
#         if self.spatial_distortion is not None:
#             positions = self.spatial_distortion(ray_samples.frustums.get_positions())
#             positions = (positions + 2.0) / 4.0
#         else:
#             positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        
#         # Mask positions that are out-of-bounds
#         selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
#         positions = positions * selector[..., None]
#         positions_flat = positions.view(-1, 3)
        
#         # Predict density values using either MLP or a single linear layer
#         if not self.use_linear:
#             density_before_activation = (
#                 self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1).to(positions)
#             )
#         else:
#             x = self.encoding(positions_flat).to(positions)
#             density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)
        
#         # Apply truncated exponential to stabilize density values
#         density = self.average_init_density * trunc_exp(density_before_activation)
        
#         # Apply threshold to convert density values to binary
#         threshold = 0.5  # This is an example value; adjust based on your needs
#         binary_density = (density > threshold).float() * self.average_init_density
        
#         # Mask out-of-bound densities
#         binary_density = binary_density * selector[..., None]
        
#         return binary_density, None
