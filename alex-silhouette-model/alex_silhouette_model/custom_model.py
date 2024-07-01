"""
Custom Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
import torch
import torch.nn as nn

from alex_silhouette_model.renderer.custom_renderers import BWRenderer
from alex_silhouette_model.loss.custom_losses import HammingDistance
from alex_silhouette_model.fields.silhouette_field import SilhouetteField

@dataclass
class CustomModelConfig(NerfactoModelConfig):
    """Custom Model Configuration.

    Add your custom model config parameters here.
    """

    use_optimized_sigmoid: bool = True
    use_weight_prioritization: bool = True
    loss_method: str = 'MSE'

    _target: Type = field(default_factory=lambda: CustomModel)


class CustomModel(NerfactoModel):
    """Custom Model."""

    config: CustomModelConfig

    def populate_modules(self):
        super().populate_modules()

        # Instead of using rgb_loss (MSELoss), use a custom binary loss (Hamming Distance)
        #self.bw_loss = HammingDistance()

        if self.config.loss_method == 'L1':
            print("--- Loss Method: L1")
            self.bw_loss = nn.L1Loss()
        else:
            print("--- Loss Method: MSE")
            self.bw_loss = nn.MSELoss()

        # Replace RGB Renderer with Custom BW Renderer
        self.renderer_bw = BWRenderer(background_color='black',
                                      use_optimized_sigmoid=self.config.use_optimized_sigmoid,
                                      use_weight_prioritization=self.config.use_weight_prioritization)

        # Custom Field instead of Nerfacto Field (Still inherited from nerfacto tho)
        # Fields

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        self.field = SilhouetteField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        # CHANGE
        #rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        bw = self.renderer_bw(weights=weights, ray_samples=ray_samples)

        # CHANGE


        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            # CHANGE
            #"rgb": rgb,
            "bw": bw,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        # CHANGE
        pred_bw, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["bw"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        # Change
        gt_r = gt_rgb[:, 0]
        pred_r = pred_bw[:, 0]
        loss_dict["bw_loss"] = self.bw_loss(gt_r, pred_r)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict


    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        # CHANGE
        # gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        predicted_bw = outputs["bw"]
        metrics_dict["psnr"] = self.psnr(predicted_bw, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        #predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)

        # CHANGE
        predicted_bw = outputs["bw"]

        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        #gt_rgb = self.renderer_rgb.blend_background(gt_rgb)

        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_bw = torch.cat([gt_rgb, predicted_bw], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        # CHANGE
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_bw = torch.moveaxis(predicted_bw, -1, 0)[None, ...]

        # CHANGE
        psnr = self.psnr(gt_rgb, predicted_bw) #predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_bw) #predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_bw) #predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        # CHANGE
        images_dict = {
            "img": combined_bw,
            "bw": predicted_bw,
            "accumulation": combined_acc,
            "depth": combined_depth
        }

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
