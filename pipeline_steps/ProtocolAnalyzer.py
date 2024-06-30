from pipeline_util.pipeline import Context
from pipeline_util.pipeline import *
from pipeline_util.utils import *

from datetime import datetime
from termcolor import cprint


class ProtocolAnalyzer:
    # def __init__(self, file_name) -> None:
    #     self._file_name = file_name

    def __call__(self, context: Context, next_step: NextStep) -> None:
        # Read the file and calculate time deltas
        with open(context.log_file_path, 'r') as file:
            lines = file.readlines()

        cprint('Analyzing Processing Time of each Step', 'white', 'on_green')

        previous_time = None

        for line in lines:
            # Log file content structure of each line: "TIMESTAMP, TEXT"
            timestamp_str, data = line.strip().split(',', 1)
            current_time = datetime.fromtimestamp(float(timestamp_str))

            if previous_time:
                delta = current_time - previous_time
                print(f"Delta: {delta} - Data: {data}")

            previous_time = current_time

        # Call the next module
        next_step(context)
