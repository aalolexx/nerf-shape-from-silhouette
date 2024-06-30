"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from silhouette_method.template_datamanager import (
    TemplateDataManagerConfig,
)
from silhouette_method.template_model import TemplateModelConfig
from silhouette_method.template_pipeline import (
    TemplatePipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

## fast and low quality configuration for quick testing ##
silhouette_method = MethodSpecification(
    config=TrainerConfig(
        method_name="silhouette-method",
        steps_per_eval_batch=100,       # Reduce from 500 to 100
        steps_per_save=500,             # Reduce from 2000 to 500
        max_num_iterations=10000,        # Reduce from 30000 to 5000
        mixed_precision=True,
        pipeline=TemplatePipelineConfig(
            datamanager=TemplateDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=1024,  # Reduce from 4096 to 1024
                eval_num_rays_per_batch=1024,   # Reduce from 4096 to 1024
            ),
            model=TemplateModelConfig(
                eval_num_rays_per_chunk=1 << 12,  # Reduce from 1 << 15 to 1 << 12
                average_init_density=0.01,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),  # Reduce from 1e-2 to 1e-3
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),  # Adjust max_steps
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),  # Reduce from 1e-2 to 1e-3
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),  # Adjust max_steps
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),  # Reduce from 1e-3 to 1e-4
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=1000),  # Adjust max_steps
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),  # Reduce from 1 << 15 to 1 << 12
        vis="viewer",
    ),
    description="Nerfstudio method for mesh out of silhouette images. !Currently using the fast/low quality version, see template_config.py!",
)

## backup of original ##

# silhouette_method = MethodSpecification(
#     config=TrainerConfig(
#         method_name="silhouette-method",
#         steps_per_eval_batch=500,
#         steps_per_save=2000,
#         max_num_iterations=30000,
#         mixed_precision=True,
#         pipeline=TemplatePipelineConfig(
#             datamanager=TemplateDataManagerConfig(
#                 dataparser=NerfstudioDataParserConfig(),
#                 train_num_rays_per_batch=4096,
#                 eval_num_rays_per_batch=4096,
#             ),
#             model=TemplateModelConfig(
#                 eval_num_rays_per_chunk=1 << 15,
#             ),
#         ),
#         optimizers={
#             # TODO: consider changing optimizers depending on your custom method
#             "proposal_networks": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
#             },
#             "fields": {
#                 "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
#             },
#             "camera_opt": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="Nerfstudio method for mesh out of silhouette images.",
# )
