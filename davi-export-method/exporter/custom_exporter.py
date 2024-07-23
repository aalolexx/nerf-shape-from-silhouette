# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for exporting NeRF into other formats.
"""


from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.scene_box import OrientedBox
# from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import collect_camera_poses, generate_point_cloud, get_mesh_from_filename
from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

# custom methods
import custom_tsdf_utils

@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        ray_bundle = RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 4 # original: 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "accumulation" # origina: rgb
    """Name of the RGB output."""
    # resolution: Union[int, List[int]] = field(default_factory=lambda: [256, 256, 256])  # original: [128, 128, 128]
    resolution: Union[int, List[int]] = field(default_factory=lambda: [64, 64, 64])  # original: [128, 128, 128]
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10  # original: 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bbox_uniform_size = 1.2
    bounding_box_min: Tuple[float, float, float] = (-bbox_uniform_size, -bbox_uniform_size, -bbox_uniform_size)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (bbox_uniform_size, bbox_uniform_size, bbox_uniform_size)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    # texture_method: Literal["tsdf", "nerf"] = "tsdf"  # original: "nerf"
    # """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    # px_per_uv_triangle: int = 1  # original: 4
    # """Number of pixels per UV triangle."""
    # unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    # """The method to use for unwrapping the mesh."""
    # num_pixels_per_side: int = 512  # original: 2048
    # """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 1000  # original: 50000
    """Target number of faces for the mesh to texture."""


    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        custom_tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )

        # # possibly
        # # texture the mesh with NeRF and export to a mesh.obj file
        # # and a material and texture file
        # if self.texture_method == "nerf":
        #     # load the mesh from the tsdf export
        #     mesh = get_mesh_from_filename(
        #         str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
        #     )
        #     CONSOLE.print("Texturing mesh with NeRF")
        #     texture_utils.export_textured_mesh(
        #         mesh,
        #         pipeline,
        #         self.output_dir,
        #         px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
        #         unwrap_method=self.unwrap_method,
        #         num_pixels_per_side=self.num_pixels_per_side,
        #     )



# @dataclass
# class ExportMarchingCubesMesh(Exporter):
#     """Export a mesh using marching cubes."""

#     isosurface_threshold: float = 0.0
#     """The isosurface threshold for extraction. For SDF based methods the surface is the zero level set."""
#     resolution: int = 1024
#     """Marching cube resolution."""
#     simplify_mesh: bool = False
#     """Whether to simplify the mesh."""
#     bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
#     """Minimum of the bounding box."""
#     bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
#     """Maximum of the bounding box."""
#     px_per_uv_triangle: int = 4
#     """Number of pixels per UV triangle."""
#     unwrap_method: Literal["xatlas", "custom"] = "xatlas"
#     """The method to use for unwrapping the mesh."""
#     num_pixels_per_side: int = 2048
#     """If using xatlas for unwrapping, the pixels per side of the texture image."""
#     target_num_faces: Optional[int] = 50000
#     """Target number of faces for the mesh to texture."""

#     def main(self) -> None:
#         """Main function."""
#         if not self.output_dir.exists():
#             self.output_dir.mkdir(parents=True)

#         _, pipeline, _, _ = eval_setup(self.load_config)

#         # TODO: Make this work with Density Field
#         assert hasattr(pipeline.model.config, "sdf_field"), "Model must have an SDF field."

#         CONSOLE.print("Extracting mesh with marching cubes... which may take a while")

#         assert self.resolution % 512 == 0, f"""resolution must be divisible by 512, got {self.resolution}.
#         This is important because the algorithm uses a multi-resolution approach
#         to evaluate the SDF where the minimum resolution is 512."""

#         # Extract mesh using marching cubes for sdf at a multi-scale resolution.
#         multi_res_mesh = generate_mesh_with_multires_marching_cubes(
#             geometry_callable_field=lambda x: cast(SDFField, pipeline.model.field)
#             .forward_geonetwork(x)[:, 0]
#             .contiguous(),
#             resolution=self.resolution,
#             bounding_box_min=self.bounding_box_min,
#             bounding_box_max=self.bounding_box_max,
#             isosurface_threshold=self.isosurface_threshold,
#             coarse_mask=None,
#         )
#         filename = self.output_dir / "sdf_marching_cubes_mesh.ply"
#         multi_res_mesh.export(filename)

#         # # load the mesh from the marching cubes export
#         # mesh = get_mesh_from_filename(str(filename), target_num_faces=self.target_num_faces)
#         # CONSOLE.print("Texturing mesh with NeRF...")
#         # texture_utils.export_textured_mesh(
#         #     mesh,
#         #     pipeline,
#         #     self.output_dir,
#         #     px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
#         #     unwrap_method=self.unwrap_method,
#         #     num_pixels_per_side=self.num_pixels_per_side,
#         # )


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        # Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
