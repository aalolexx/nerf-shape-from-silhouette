from pipeline_util.pipeline import Context
from pipeline_util.pipeline import *
from pipeline_util.utils import *

from termcolor import cprint
import shutil
import os


class FileCopier:
    def __init__(self, file_from, file_to) -> None:
        self._file_from = file_from
        self._file_to = file_to

    def __call__(self, context: Context, next_step: NextStep) -> None:

        if not os.path.exists(os.path.dirname(self._file_to)):
            os.makedirs(os.path.dirname(self._file_to))

        try:
            # Move the file or directory
            shutil.copy(self._file_from, self._file_to)
            cprint("Copied file " + self._file_from + " to " + self._file_to, "green",)
        except Exception as e:
            cprint("An error occurred: " + str(e), "red")


        # Call the next module
        next_step(context)
