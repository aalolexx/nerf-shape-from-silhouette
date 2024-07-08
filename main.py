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
def run_pipeline_images_realworld():
    ctx = get_new_context()
    pipeline_pie = Pipeline[PipeContext](
        Protocoller("starting pipeline"),
        DepthEstimator("data/input/dnerf/hook/test", "data/working/depth_images/hook/test", threshold=110),
        DepthEstimator("data/input/dnerf/hook/train", "data/working/depth_images/hook/train", threshold=110),
        DepthEstimator("data/input/dnerf/hook/val", "data/working/depth_images/hook/val", threshold=110),
        #ImageBinarizerFacebook("data/input/test-images/", "data/working/depth_images/", "data/working/binarized_images/")
        ImageBinarizerRMBG("data/input/dnerf/hook/test", "data/working/depth_images/hook/test", "data/working/binarized_images/hook/test"),
        ImageBinarizerRMBG("data/input/dnerf/hook/train", "data/working/depth_images/hook/train", "data/working/binarized_images/hook/train"),
        ImageBinarizerRMBG("data/input/dnerf/hook/val", "data/working/depth_images/hook/val", "data/working/binarized_images/hook/val")
    )
    pipeline_pie(ctx, error_handler)


def run_pipeline_images_synthetic():
    ctx = get_new_context()
    pipeline_pie = Pipeline[PipeContext](
        Protocoller("started pipeline"),
        ImageBinarizerRMBGNoDepth("data/input/dnerf/hook/test", "data/working/binarized_images_lowres/hook/test", target_width=256),
        ImageBinarizerRMBGNoDepth("data/input/dnerf/hook/train", "data/working/binarized_images_lowres/hook/train", target_width=256),
        ImageBinarizerRMBGNoDepth("data/input/dnerf/hook/val", "data/working/binarized_images_lowres/hook/val", target_width=256),

    )
    pipeline_pie(ctx, error_handler)


def run_pipeline_test_loss_methods():
    input_data_path = "data/input/dnerf-hook-bin"
    ctx = get_new_context()
    pipeline_pie = Pipeline[PipeContext](
        Protocoller("prepared dataset"),
        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               loss_method="MSE",
                               export_postfix="_MSE"),
        Protocoller("trained alex-silhouette-model MSE"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               loss_method="L1",
                               export_postfix="_L1"),
        Protocoller("trained alex-silhouette-model L1"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               loss_method="SmoothL1",
                               export_postfix="_SmoothL1"),
        Protocoller("trained alex-silhouette-model SmoothL1"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               loss_method="KLDiv",
                               export_postfix="_KLDiv"),
        Protocoller("trained alex-silhouette-model KLDiv"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               loss_method="Charbonnier",
                               export_postfix="_Charbonnier"),
        Protocoller("trained alex-silhouette-model Charbonnier"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               loss_method="BCEWithLogits",
                               export_postfix="_BCEWithLogits"),
        Protocoller("trained alex-silhouette-model BCEWithLogits"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               loss_method="Tversky",
                               export_postfix="_Tversky"),
        Protocoller("trained alex-silhouette-model Tversky"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=True,
                               use_weight_prioritization=True,
                               loss_method="MSE",
                               export_postfix="_MSE_os_wp"),
        Protocoller("trained alex-silhouette-model MSE oSig+wPrio"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=True,
                               use_weight_prioritization=True,
                               loss_method="L1",
                               export_postfix="_L1_os_wp"),
        Protocoller("trained alex-silhouette-model L1 oSig+wPrio"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=True,
                               use_weight_prioritization=True,
                               loss_method="SmoothL1",
                               export_postfix="_SmoothL1_os_wp"),
        Protocoller("trained alex-silhouette-model SmoothL1 oSig+wPrio"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=True,
                               use_weight_prioritization=True,
                               loss_method="KLDiv",
                               export_postfix="_KLDiv_os_wp"),
        Protocoller("trained alex-silhouette-model KLDiv oSig+wPrio"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=True,
                               use_weight_prioritization=True,
                               loss_method="Charbonnier",
                               export_postfix="_Charbonnier_os_wp"),
        Protocoller("trained alex-silhouette-model Charbonnier oSig+wPrio"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=True,
                               use_weight_prioritization=True,
                               loss_method="BCEWithLogits",
                               export_postfix="_BCEWithLogits_os_wp"),
        Protocoller("trained alex-silhouette-model BCEWithLogits oSig+wPrio"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=True,
                               use_weight_prioritization=True,
                               loss_method="Tversky",
                               export_postfix="_Tversky_os_wp"),
        Protocoller("trained alex-silhouette-model Tversky oSig+wPrio"),

        ProtocolAnalyzer(),
    )
    pipeline_pie(ctx, error_handler)


def run_pipeline_test_sig_params():
    input_data_path = "data/input/dnerf-hook-bin"
    ctx = get_new_context()
    pipeline_pie = Pipeline[PipeContext](
        Protocoller("prepared dataset"),
        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               renderer_sig_range=5.0,
                               renderer_sig_offset=0.0,
                               loss_method="L1",
                               export_postfix="_l1_r5_o0",
                               ),
        Protocoller("trained alex-silhouette-model L1. range=5.0, offset=4.0"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               renderer_sig_range=10.0,
                               renderer_sig_offset=0.0,
                               loss_method="L1",
                               export_postfix="_l1_r10_o0",
                               ),
        Protocoller("trained alex-silhouette-model L1. range=10.0, offset=0.0"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               renderer_sig_range=15.0,
                               renderer_sig_offset=0.0,
                               loss_method="L1",
                               export_postfix="_l1_r15_o0",
                               ),
        Protocoller("trained alex-silhouette-model L1. range=15.0, offset=0.0"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               renderer_sig_range=5.0,
                               renderer_sig_offset=5.0,
                               loss_method="L1",
                               export_postfix="_l1_r5_o5",
                               ),
        Protocoller("trained alex-silhouette-model L1. range=5.0, offset=5.0"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               renderer_sig_range=10.0,
                               renderer_sig_offset=5.0,
                               loss_method="L1",
                               export_postfix="_l1_r10_o5",
                               ),
        Protocoller("trained alex-silhouette-model L1. range=10.0, offset=5.0"),

        NerfstudioTrainStarter(model="alex-silhouette-model",
                               data_path=input_data_path,
                               use_optimized_sigmoid=False,
                               use_weight_prioritization=False,
                               renderer_sig_range=15.0,
                               renderer_sig_offset=5.0,
                               loss_method="L1",
                               export_postfix="_l1_r15_o5",
                               ),
        Protocoller("trained alex-silhouette-model L1. range=15.0, offset=5.0"),



        ProtocolAnalyzer(),
    )
    pipeline_pie(ctx, error_handler)


if __name__ == "__main__":
    run_pipeline_test_sig_params()
