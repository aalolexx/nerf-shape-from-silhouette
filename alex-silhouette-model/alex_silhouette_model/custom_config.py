from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from alex_silhouette_model.custom_datamanager import (
    CustomDataManagerConfig,
)
from alex_silhouette_model.custom_model import CustomModelConfig
from alex_silhouette_model.custom_pipeline import (
    CustomPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

alex_silhouette_model = MethodSpecification(
  config=TrainerConfig(
    method_name="alex-silhouette-model",
    pipeline=CustomPipelineConfig(
        datamanager=CustomDataManagerConfig(
        dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
            model=CustomModelConfig(
            eval_num_rays_per_chunk=1 << 15,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
  ),
  description="Custom description"
)