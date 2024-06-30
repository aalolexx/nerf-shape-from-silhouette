from pipeline_util.context import PipeContext
from pipeline_util.pipeline import *
#from pipeline_steps.ImageBinarizerFacebook import ImageBinarizerFacebook
from pipeline_steps.ImageBinarizerRMBG import ImageBinarizerRMBG
from pipeline_steps.ImageBinarizerRMBGNoDepth import ImageBinarizerRMBGNoDepth
from pipeline_steps.DepthEstimator import DepthEstimator
from pipeline_steps.Protocoller import Protocoller
from pipeline_steps.ProtocolAnalyzer import (ProtocolAnalyzer)
from pipeline_steps.NerfstudioTrainStarter import NerfstudioTrainStarter

import time

#
# Pipeline Preparations
#
# TODO actually use these path variables
def get_new_context():
    return PipeContext(
        input_dir_path="data/input",
        output_dir_path="data/output",
        working_dir_path="data/working",
        misc_dir_path="data/misc",
        log_file_path="data/output/run_" + str(time.time()) + "_log.txt"
    )


def error_handler(error: Exception, context: Context, next_step: NextStep):
    # TODO Make Error Handler more sophisticated
    print(error)
    raise ValueError(error) from error


#
# Run actual Pipelines
#
"""
def run_pipeline_images_realworld():
    ctx = get_new_context()
    pipeline_pie = Pipeline[PipeContext](
        Protocoller('starting pipeline'),
        DepthEstimator('data/input/dnerf/hook/test', 'data/working/depth_images/hook/test', threshold=110),
        DepthEstimator('data/input/dnerf/hook/train', 'data/working/depth_images/hook/train', threshold=110),
        DepthEstimator('data/input/dnerf/hook/val', 'data/working/depth_images/hook/val', threshold=110),
        #ImageBinarizerFacebook('data/input/test-images/', 'data/working/depth_images/', 'data/working/binarized_images/')
        ImageBinarizerRMBG('data/input/dnerf/hook/test', 'data/working/depth_images/hook/test', 'data/working/binarized_images/hook/test'),
        ImageBinarizerRMBG('data/input/dnerf/hook/train', 'data/working/depth_images/hook/train', 'data/working/binarized_images/hook/train'),
        ImageBinarizerRMBG('data/input/dnerf/hook/val', 'data/working/depth_images/hook/val', 'data/working/binarized_images/hook/val')
    )
    pipeline_pie(ctx, error_handler)
"""

def run_pipeline_images_synthetic():
    ctx = get_new_context()
    pipeline_pie = Pipeline[PipeContext](
        Protocoller('started pipeline'),
        ImageBinarizerRMBGNoDepth('data/input/dnerf/hook/test', 'data/working/binarized_images_lowres/hook/test', target_width=256),
        ImageBinarizerRMBGNoDepth('data/input/dnerf/hook/train', 'data/working/binarized_images_lowres/hook/train', target_width=256),
        ImageBinarizerRMBGNoDepth('data/input/dnerf/hook/val', 'data/working/binarized_images_lowres/hook/val', target_width=256),

    )
    pipeline_pie(ctx, error_handler)


def run_pipeline_silhouette_method():
    ctx = get_new_context()
    pipeline_pie = Pipeline[PipeContext](
        Protocoller('prepared dataset'),
        NerfstudioTrainStarter('nerfacto', 'data/input/dnerf-hook-bin'),
        Protocoller('trained nerfacto'),
        NerfstudioTrainStarter('alex-silhouette-model', 'data/input/dnerf-hook-bin'),
        Protocoller('trained alex-silhouette-model'),
        ProtocolAnalyzer(),
    )
    pipeline_pie(ctx, error_handler)

if __name__ == '__main__':
    run_pipeline_silhouette_method()
