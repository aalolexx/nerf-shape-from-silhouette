from pipeline_util.context import PipeContext
from pipeline_util.pipeline import *
from pipeline_steps.ImageBinarizer import ImageBinarizer
from pipeline_steps.DepthEstimator import DepthEstimator


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
        DepthEstimator('data/input/blendMVS/blended_images/', 'data/working/depth_images/', threshold=135),
        ImageBinarizer('data/input/blendMVS/blended_images/', 'data/working/depth_images/', 'data/working/binarized_images/')
    )
    pipeline_pie(ctx, error_handler)


if __name__ == '__main__':
    run_demo_pipeline()
