from pipeline_util.context import PipeContext
from pipeline_util.pipeline import *
from pipeline_steps.ImageBinarizerRMBGNoDepth import ImageBinarizerRMBGNoDepth
from pipeline_steps.Protocoller import Protocoller
from pipeline_steps.ProtocolAnalyzer import (ProtocolAnalyzer)
from pipeline_steps.NerfstudioTrainStarter import NerfstudioTrainStarter
from pipeline_steps.FileCopier import FileCopier
from pipeline_steps.MeshExporter import MeshExporter

import time

#
# Pipeline Preparations
#
def get_new_context():
    return PipeContext(
        input_dir_path="data/input",
        output_dir_path="data/output",
        working_dir_path="data/working",
        misc_dir_path="data/misc",
        log_file_path="data/output/run_" + str(time.time()) + "_log.txt",
        current_run_export_dir="" # Will be set by trainer
    )


def error_handler(error: Exception, context: Context, next_step: NextStep):
    # TODO Make Error Handler more sophisticated
    print(error)
    raise ValueError(error) from error


#
# Pipeline to fully run the whole process for a dnerf_data dataset
# Download Dnerf Data using ns-download command
#
def run_demo_pipeline_with_dnerf_data(dataset):
    dataset_name = dataset
    model_name = "alex-silhouette-model"
    input_data_path = "data/working/binarized_images_lowres/" + dataset_name
    export_config_base_dir = "outputs/" + dataset_name + "/" + model_name

    ctx = get_new_context()

    pipeline_pie = Pipeline[PipeContext](
        Protocoller("started pipeline"),

        # Run Segmentation Steps
        ImageBinarizerRMBGNoDepth("data/input/dnerf/" + dataset_name + "/test",
                                  "data/working/binarized_images_lowres/" + dataset_name + "/test",
                                  target_width=600),
        ImageBinarizerRMBGNoDepth("data/input/dnerf/" + dataset_name + "/train",
                                  "data/working/binarized_images_lowres/" + dataset_name + "/train",
                                  target_width=600),
        ImageBinarizerRMBGNoDepth("data/input/dnerf/" + dataset_name + "/val",
                                  "data/working/binarized_images_lowres/" + dataset_name + "/val",
                                  target_width=600),

        ## Move Camera Meta Files (DNERF specific)
        FileCopier("data/input/dnerf/" + dataset_name + "/transforms_train.json",
                   "data/working/binarized_images_lowres/" + dataset_name + "/transforms_train.json"),
        FileCopier("data/input/dnerf/" + dataset_name + "/transforms_test.json",
                   "data/working/binarized_images_lowres/" + dataset_name + "/transforms_test.json"),
        FileCopier("data/input/dnerf/" + dataset_name + "/transforms_val.json",
                   "data/working/binarized_images_lowres/" + dataset_name + "/transforms_val.json"),
        Protocoller("prepared dataset"),

        # Actual NERF Training
        NerfstudioTrainStarter(model=model_name,
                               experiment_name=dataset_name,
                               data_path=input_data_path,
                               use_optimized_sigmoid=True,
                               use_weight_prioritization=False,
                               renderer_sig_range=10.0,
                               renderer_sig_offset=5.0,
                               loss_method="L1",
                               export_postfix="_l1_oSig_" + dataset,
                               ),
        Protocoller("trained alex-silhouette-model L1. range=15.0, offset=5.0"),

        # Export the run

        MeshExporter(
            export_config_base_dir=export_config_base_dir,
            export_type="tsdf",  # Choose the type of export: tsdf, poisson, marching-cubes
            rgb_output_name="bw",
        ),
        ProtocolAnalyzer(),
    )
    pipeline_pie(ctx, error_handler)


#
# Pipeline to fully run the whole process for a custom real world dataset
# Contact the repository maintainers to get access to this dataset
#
def run_demo_pipeline_with_camera_data():
    dataset_name = "wall-e"
    model_name = "alex-silhouette-model"
    input_data_path = "data/working/binarized_images_origres/" + dataset_name
    export_config_base_dir = "outputs/" + dataset_name + "/" + model_name

    ctx = get_new_context()

    pipeline_pie = Pipeline[PipeContext](
        Protocoller("started pipeline"),

        # Run Segmentation Steps
        ImageBinarizerRMBGNoDepth("data/input/" + dataset_name + "/images",
                                  "data/working/binarized_images_origres/" + dataset_name + "/images"),
        ImageBinarizerRMBGNoDepth("data/input/" + dataset_name + "/images_2",
                                  "data/working/binarized_images_origres/" + dataset_name + "/images_2"),
        ImageBinarizerRMBGNoDepth("data/input/" + dataset_name + "/images_4",
                                  "data/working/binarized_images_origres/" + dataset_name + "/images_4"),
        ImageBinarizerRMBGNoDepth("data/input/" + dataset_name + "/images_8",
                                  "data/working/binarized_images_origres/" + dataset_name + "/images_8"),

        # Move Camera Meta Files (DNERF specific)
        FileCopier("data/input/" + dataset_name + "/colmap",
                   "data/working/binarized_images_origres/" + dataset_name + "/colmap"),
        FileCopier("data/input/" + dataset_name + "/sparse_pc.ply",
                   "data/working/binarized_images_origres/" + dataset_name + "/sparse_pc.ply"),
        FileCopier("data/input/" + dataset_name + "/transforms.json",
                   "data/working/binarized_images_origres/" + dataset_name + "/transforms.json"),
        Protocoller("prepared dataset"),

        # Actual NERF Training
        NerfstudioTrainStarter(model=model_name,
                               experiment_name=dataset_name,
                               data_path=input_data_path,
                               dataparser="nerfacto",
                               use_optimized_sigmoid=True,
                               use_weight_prioritization=False,
                               renderer_sig_range=10.0,
                               renderer_sig_offset=5.0,
                               loss_method="L1",
                               export_postfix="_l1_oSig_" + dataset_name,
                               ),
        Protocoller("trained alex-silhouette-model L1. range=15.0, offset=5.0"),

        # Export the run
        MeshExporter(
            export_config_base_dir=export_config_base_dir,
            export_type="tsdf",  # Choose the type of export: tsdf, poisson, marching-cubes
            rgb_output_name="bw",
        ),
        ProtocolAnalyzer(),
    )
    pipeline_pie(ctx, error_handler)


if __name__ == "__main__":
    #run_demo_pipeline_with_dnerf_data("hook")
    #run_demo_pipeline_with_dnerf_data("lego")
    #run_demo_pipeline_with_dnerf_data("trex")
    run_demo_pipeline_with_camera_data()
