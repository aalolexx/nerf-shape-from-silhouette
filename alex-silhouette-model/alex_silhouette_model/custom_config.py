from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from alex_silhouette_model.custom_datamanager import (
    CustomDataManagerConfig,
)
from alex_silhouette_model.custom_dataparser import (
    CustomDataParserConfig,
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
    timestamp="{timestamp}",
    steps_per_eval_batch=100,  # 500,
    steps_per_save=500,  # 2000,
    max_num_iterations=5000,  # 30000,
    mixed_precision=True,
    pipeline=CustomPipelineConfig(
        datamanager=CustomDataManagerConfig(
            dataparser=CustomDataParserConfig(),
            train_num_rays_per_batch=1024,  # 4096,
            eval_num_rays_per_batch=1024,  # 4096,
        ),
        model=CustomModelConfig(
            eval_num_rays_per_chunk=1 << 12,  # 15
            average_init_density=0.01,
            # Our newly introduced options
            use_optimized_sigmoid=True,
            use_weight_prioritization=False,
            renderer_sig_range=10.0,
            renderer_sig_offset=5.0,
            loss_method='L1',
            predict_normals=True,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
        },
        "camera_opt": {
           "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
        },
    },
    viewer=ViewerConfig(
        #num_rays_per_chunk=1 << 12, # 15
        quit_on_train_completion=True
    ),
    vis="viewer",
  ),
  description="Custom description"
)