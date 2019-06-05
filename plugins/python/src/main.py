import os.path
import sys
import supervisely_lib as sly
from supervisely_lib.task import paths as sly_paths

sys.path.append(os.path.join(sly_paths.TaskPaths.TASK_DIR, sly_paths.CODE))

def main():
    import script


if __name__ == '__main__':
    sly.main_wrapper('PYTHON_TASK', main)
