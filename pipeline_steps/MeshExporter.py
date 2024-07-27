from pipeline_util.pipeline import Context, NextStep
from pathlib import Path
from datetime import datetime
import sys
import subprocess

# Add the path to the exporter module
sys.path.append('davi-export-method')
from exporter import custom_exporter
# from exporter import custom_tsdf_utils

class MeshExporter:
    def __init__(self, export_config_base_dir: str, export_type: str = "tsdf", **kwargs) -> None:
        self.export_config_base_dir = export_config_base_dir
        self.export_type = export_type
        self.export_params = kwargs

    def __call__(self, context: Context, next_step: NextStep) -> None:
        # Print the export command (for debugging purposes)
        command = f"Custom export using {self.export_type} method"
        print(command)

        # Set up the export configuration
        config = custom_exporter.ExportTSDFMesh if self.export_type == "tsdf" else \
                 custom_exporter.ExportPoissonMesh if self.export_type == "poisson" else \
                 custom_exporter.ExportMarchingCubesMesh

        # Create the exporter object with the necessary parameters
        export_config_path = Path(self.export_config_base_dir + "/" + context.current_run_export_dir + "/config.yml")
        output_dir = Path(context.output_dir_path + "/" + context.current_run_export_dir)

        exporter = config(load_config=export_config_path, output_dir=output_dir, **self.export_params)

        # Call the main export function
        exporter.main()

        # Call the next step in the pipeline
        next_step(context)