from pipeline_util.pipeline import Context
from pipeline_util.pipeline import *
from pipeline_util.utils import *

import subprocess

from nerfstudio.scripts.train import main
from nerfstudio.engine import trainer
from termcolor import cprint
from pathlib import Path

import sys
sys.path.append('../alex-silhouette-model')
from alex_silhouette_model import custom_config

class NerfstudioTrainStarter:
    def __init__(self, model: str,
                 data_path: str) -> None:
        self._model = model
        self._data_path = data_path

    def __call__(self, context: Context, next_step: NextStep) -> None:

        command = f"ns-train {self._model} --data {self._data_path} --vis tensorboard"
        cprint(command, "yellow")

        conf = custom_config.alex_silhouette_model.config
        data_path = Path(self._data_path)
        conf.method_name = self._model
        conf.pipeline.datamanager.data = data_path
        conf.vis = 'tensorboard'
        main(conf)

        # Call the next module
        next_step(context)
