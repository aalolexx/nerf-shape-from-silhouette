from pipeline_util.pipeline import Context
from pipeline_util.pipeline import *


class DemoStep1:
    # Define custom step/module parameters here (only usable in this module, not in the whole pipeline)
    def __init__(self, custom_msg: str) -> None:
        self._custom_msg = custom_msg

    # The Method that's called by the pipeline.
    # Do the actual step/module code here and call the next step/module as soon as you're done
    # The Context Var can be used/modified in this method to pass data to the next steps/modules
    def __call__(self, context: Context, next_step: NextStep) -> None:
        print("Hello from Demo Step 1: " + str(self._custom_msg))
        complicated_module_calculation = 1 + 2
        context.demo_context_var = complicated_module_calculation
        # Call the next module
        next_step(context)
