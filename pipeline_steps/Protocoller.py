from pipeline_util.pipeline import Context
from pipeline_util.pipeline import *
from pipeline_util.utils import *

import time
from termcolor import cprint


class Protocoller:
    def __init__(self, protocol_text: str) -> None:
        self._protocol_text = protocol_text

    def __call__(self, context: Context, next_step: NextStep) -> None:
        ts = time.time()
        log_entry = f"{ts}, {self._protocol_text}"

        with open(context.log_file_path, "a") as file:
            file.write(log_entry + '\n')

        cprint(f"Logged {self._protocol_text}")

        # Call the next module
        next_step(context)
