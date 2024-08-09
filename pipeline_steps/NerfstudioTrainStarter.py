import nerfstudio.data.dataparsers.nerfstudio_dataparser

from pipeline_util.pipeline import Context
from pipeline_util.pipeline import *
from pipeline_util.utils import *

import subprocess

from nerfstudio.scripts.train import main
from termcolor import cprint
from pathlib import Path
from datetime import datetime
import secrets

import sys
sys.path.append('../alex-silhouette-model')
from alex_silhouette_model import custom_config

class NerfstudioTrainStarter:
    def __init__(self, model: str,
                 data_path: str,
                 experiment_name: str = "default_experiment",
                 use_optimized_sigmoid: bool = True,
                 use_weight_prioritization: bool = True,
                 renderer_sig_range: float = 8.4,
                 renderer_sig_offset: float = 0.0,
                 loss_method: str = "MSE",
                 export_postfix: str = "",
                 max_num_iterations: int = 10000,
                 steps_per_save: int = 500,
                 steps_per_eval_batch: int = 100,
                 vis: str = "viewer",
                 dataparser: str = "dnerf") -> None:
        self._model = model
        self._experiment_name = experiment_name
        self._data_path = data_path
        self._use_optimized_sigmoid = use_optimized_sigmoid
        self._use_weight_prioritization = use_weight_prioritization,
        self._renderer_sig_range = renderer_sig_range
        self._renderer_sig_offset = renderer_sig_offset
        self._loss_method = loss_method
        self._max_num_iterations = max_num_iterations
        self._steps_per_save = steps_per_save
        self._steps_per_eval_batch = steps_per_eval_batch
        self._export_postfix = export_postfix
        self._vis = vis
        self._dataparser = dataparser

    def __call__(self, context: Context, next_step: NextStep) -> None:

        command = f"ns-train {self._model} --data {self._data_path} --vis " + self._vis
        cprint(command, "yellow")

        run_export_dir = datetime.now().strftime("%Y-%m-%d_%H%M%S") + self._export_postfix
        conf = custom_config.alex_silhouette_model.config
        data_path = Path(self._data_path)

        conf.method_name = self._model
        conf.experiment_name = self._experiment_name
        conf.timestamp = run_export_dir
        conf.steps_per_eval_batch = self._steps_per_eval_batch
        conf.steps_per_save = self._steps_per_save
        conf.max_num_iterations = self._max_num_iterations
        conf.vis = self._vis
        conf.pipeline.datamanager.data = data_path
        conf.pipeline.model.use_optimized_sigmoid = self._use_optimized_sigmoid
        conf.pipeline.model.use_weight_prioritization = self._use_weight_prioritization
        conf.pipeline.model.loss_method = self._loss_method
        conf.pipeline.model.renderer_sig_range = self._renderer_sig_range
        conf.pipeline.model.renderer_sig_offset = self._renderer_sig_offset

        # Default is DNERF and it's already set up in the default config of the method
        # only one besides DNERF supported by our pipeline right now is nerfacto
        # -> use this one for real world datasets processed with colmap (ns-process)
        if self._dataparser == "nerfacto":
            conf.pipeline.datamanager.dataparser = nerfstudio.data.dataparsers.nerfstudio_dataparser.NerfstudioDataParserConfig()

        main(conf)

        context.current_run_export_dir = run_export_dir

        # Call the next module
        next_step(context)
