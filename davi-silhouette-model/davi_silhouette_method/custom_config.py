from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from davi_silhouette_method.custom_datamanager import (
    CustomDataManagerConfig,
)
from davi_silhouette_method.custom_dataparser import (
    CustomDataParserConfig,
)
from davi_silhouette_method.custom_model import CustomModelConfig
from davi_silhouette_method.custom_pipeline import (
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

## fast and low quality configuration for quick testing ##
davi_silhouette_model = MethodSpecification(
    config=TrainerConfig(
        method_name="davi-silhouette-method",
        steps_per_eval_batch=100,       # Reduce from 500 to 100
        steps_per_save=500,             # Reduce from 2000 to 500
        max_num_iterations=5000,        # Reduce from 30000 to 5000
        mixed_precision=True,
        pipeline=CustomPipelineConfig(
            datamanager=CustomDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=1024,  # Reduce from 4096 to 1024
                eval_num_rays_per_batch=1024,   # Reduce from 4096 to 1024
                camera_res_scale_factor=0.5,
            ),
            model=CustomModelConfig(
                eval_num_rays_per_chunk=1 << 15,  # Reduce from 1 << 15 to 1 << 12
                # for B/W
                # average_init_density=.01,
                disable_scene_contraction=True,
                average_init_density=.5,
                background_color="black",
                # some more for quick rendering:
                base_res=8,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
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
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),  # Reduce from 1 << 15 to 1 << 12
        vis="viewer",#"tensorboard",
    ),
    description="Nerfstudio method for mesh out of silhouette images. !Currently using the fast/low quality version, see Custom_config.py!",
)
