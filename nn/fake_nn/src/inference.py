# coding: utf-8

import cv2
import time
import os.path as osp

from supervisely_lib import logger
import supervisely_lib as sly


def load_fake_model(ckpt_dir):
    with open(osp.join(ckpt_dir, 'weights.bin'), 'r') as f:
        res = f.read()
    return res


def main():
    # Please note that auxiliary methods from sly (supervisely_lib) use supervisely_lib.logger to format output.
    # So don't replace formatters or handlers of the logger.
    # One may use other loggers or simple prints for other output, but it's recommended to use supervisely_lib.logger.
    logger.info('Hello ML world')
    print('Glad to see u')

    # TaskHelperTrain contains almost all needed to run inference as Supervisely task,
    # including task settings and paths to data and model.
    task_helper = sly.TaskHelperInference()

    # All settings and parameters are passed to task in json file.
    # Content of the file is entirely dependent on model implementation.
    inference_settings = task_helper.task_settings
    logger.info('Task settings are read', extra={'task_settings': inference_settings})

    # Let's imitate loading training model weights. Task acquires directory with the weights.
    # And content of the directory is entirely dependent on model implementation.
    model_dir = task_helper.paths.model_dir
    model = load_fake_model(model_dir)
    logger.info('Model weights are loaded', extra={'model_dir': model_dir})

    # We are going to read input project with data for inference.
    project_meta = task_helper.in_project_meta  # Project meta contains list of project classes.
    project_dir = task_helper.paths.project_dir
    project_fs = sly.ProjectFS.from_disk_dir_project(project_dir)
    # ProjectFS enlists all samples (image/annotation pairs) in input project.

    # We are to write inference results as sly project with same structure into provided results dir.
    # There is no need to save images, only annotations and project meta are required.
    results_dir = task_helper.paths.results_dir
    out_project_fs = sly.ProjectFS(results_dir, project_fs.pr_structure)

    # It's necessary to write project meta (with list of classes) for output project.
    out_meta = sly.ProjectMeta([{'title': 'hat', 'shape': 'point', 'color': '#FF0000'}])  # create meta
    out_meta.to_dir(out_project_fs.project_path)  # and save

    # We are to report progress of task over sly.ProgressCounter if we want to observe the progress in web panel.
    # In fact one task may report progress for some sequential (not nested) subtasks,
    # but here we will report inference progress only.
    ia_cnt = out_project_fs.pr_structure.image_cnt
    progress = sly.progress_counter_inference(cnt_imgs=ia_cnt)

    # Iterating over samples (image/annotation pairs) in input project.
    for item_descr in project_fs:
        logger.info('Processing input sample',
                    extra={'dataset': item_descr.ds_name, 'image_name': item_descr.image_name})

        # Open some image...
        img = cv2.imread(item_descr.img_path)
        logger.info('Read image from input project',
                    extra={'width': img.shape[1], 'height': img.shape[0]})

        # And read corresponding annotation...
        ann_packed = sly.json_load(item_descr.ann_path)
        ann = sly.Annotation.from_packed(ann_packed, project_meta)
        logger.info('Read annotation from input project', extra={'tags': ann['tags']})

        logger.info('Some forward pass...')
        time.sleep(1)

        # Let's imitate inference output.
        # We are to save results as sly Figures in annotation.
        ann['objects'] = [sly.FigurePoint.from_pt('hat', (800, 800))]  # imagine that our model found the point
        out_ann_path = out_project_fs.ann_path(item_descr.ds_name, item_descr.image_name)
        sly.ensure_base_path(out_ann_path)  # create intermediate directories
        sly.json_dump(ann.pack(), out_ann_path)  # and save annotation
        # Note that there is no need to save image.

        progress.iter_done_report()  # call it after every iteration to report progress

    # It's necessary to report that the inference task is finished.
    sly.report_inference_finished()

    # Thank you for your patience.
    logger.info('Applying finished.')


if __name__ == '__main__':
    main()
