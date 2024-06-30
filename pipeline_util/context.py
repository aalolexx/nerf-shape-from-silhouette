from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PipeContext:
    input_dir_path: str
    output_dir_path: str
    working_dir_path: str
    misc_dir_path: str
    log_file_path: str
