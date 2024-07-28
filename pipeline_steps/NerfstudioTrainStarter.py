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
                 export_postfix: str = "") -> None:
        self._model = model
        self._experiment_name = experiment_name
        self._data_path = data_path
        self._use_optimized_sigmoid = use_optimized_sigmoid
        self._use_weight_prioritization = use_weight_prioritization,
        self._renderer_sig_range = renderer_sig_range
        self._renderer_sig_offset = renderer_sig_offset
        self._loss_method = loss_method
        self._export_postfix = export_postfix

    def __call__(self, context: Context, next_step: NextStep) -> None:

        command = f"ns-train {self._model} --data {self._data_path} --vis viewer"
        cprint(command, "yellow")

        run_export_dir = datetime.now().strftime("%Y-%m-%d_%H%M%S") + self._export_postfix
        conf = custom_config.alex_silhouette_model.config
        data_path = Path(self._data_path)

        conf.method_name = self._model
        conf.experiment_name = self._experiment_name
        conf.timestamp = run_export_dir
        conf.pipeline.datamanager.data = data_path
        conf.pipeline.model.use_optimized_sigmoid = self._use_optimized_sigmoid
        conf.pipeline.model.use_weight_prioritization = self._use_weight_prioritization
        conf.pipeline.model.loss_method = self._loss_method
        conf.pipeline.model.renderer_sig_range = self._renderer_sig_range
        conf.pipeline.model.renderer_sig_offset = self._renderer_sig_offset

        main(conf)

        context.current_run_export_dir = run_export_dir

        # Call the next module
        next_step(context)
