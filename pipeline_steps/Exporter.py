from pipeline_util.pipeline import Context, NextStep
from pathlib import Path
from datetime import datetime
import sys
import subprocess

# Add the path to the exporter module
sys.path.append('davi-export-method')
from exporter import custom_exporter
# from exporter import custom_tsdf_utils

class Exporter:
    def __init__(self, load_config: str, output_dir: str, export_type: str = "tsdf", **kwargs) -> None:
        self.load_config = Path(load_config)
        self.output_dir = Path(output_dir)
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
        exporter = config(load_config=self.load_config, output_dir=self.output_dir, **self.export_params)

        # Call the main export function
        exporter.main()

        # Call the next step in the pipeline
        next_step(context)

# # Example usage
# if __name__ == "__main__":
#     # Example call to the CustomExporter class
#     custom_exporter = CustomExporter(
#         load_config="path/to/config.yml",
#         output_dir="path/to/output",
#         export_type="tsdf",  # or "poisson" or "marching-cubes"
#         rgb_output_name="bw",
#         # Add other parameters specific to the export type
#     )

#     # Call the exporter with a mock context and next_step
#     custom_exporter(Context(), lambda x: print("Next step"))
