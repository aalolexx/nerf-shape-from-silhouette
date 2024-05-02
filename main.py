from pipeline_util.context import PipeContext
from pipeline_util.pipeline import *
from pipeline_steps.demo.DemoStep1 import DemoStep1
from pipeline_steps.demo.DemoStep2 import DemoStep2


#
# Pipeline Preparations
#
def get_new_context():
    return PipeContext(
        input_dir_path='data/input',
        output_dir_path='data/output',
        working_dir_path='data/working',
        misc_dir_path='data/misc',
        demo_context_var=0
    )


def error_handler(error: Exception, context: Context, next_step: NextStep):
    # TODO Make Error Handler more sophisticated
    print(error)
    raise ValueError(error) from error


#
# Run actual Pipeline
#
def run_demo_pipeline():
    ctx = get_new_context()
    pipeline_pie = Pipeline[PipeContext](
        DemoStep1(custom_msg='I am msg 1'),
        DemoStep2(custom_msg='msg 2!')
    )
    pipeline_pie(ctx, error_handler)


if __name__ == '__main__':
    run_demo_pipeline()
