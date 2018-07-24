# coding: utf-8

import time
import os.path as osp
import random

import cv2
from supervisely_lib import logger
import supervisely_lib as sly


def create_fake_model():
    return 'FAKE_MODEL_WEIGHTS'


def load_fake_model(ckpt_dir):
    with open(osp.join(ckpt_dir, 'weights.bin'), 'r') as f:
        res = f.read()
    return res


def dump_fake_model(ckpt_dir, model):
    with open(osp.join(ckpt_dir, 'weights.bin'), 'w') as f:
        f.write(model)


def main():
    # Please note that auxiliary methods from sly (supervisely_lib) use supervisely_lib.logger to format output.
    # So don't replace formatters or handlers of the logger.
    # One may use other loggers or simple prints for other output, but it's recommended to use supervisely_lib.logger.
    logger.info('Hello ML world')
    print('Glad to see u')

    # TaskHelperTrain contains almost all needed to run training as Supervisely task,
    # including task settings and paths to data and models.
    task_helper = sly.TaskHelperTrain()

    # All settings and parameters are passed to task in json file.
    # Content of the file is entirely dependent on model implementation.
    training_settings = task_helper.task_settings
    logger.info('Task settings are read', extra={'task_settings': training_settings})
    cnt_epochs = training_settings["epochs"]  # in the fake model we want cnt of epochs
    cnt_iters_per_epoch = training_settings["iters_per_epoch"]

    # Let's imitate model weights loading.
    # Task acquires directory with input model weights (e.g. to continue training or to initialize some parameters).
    # Content of the directory is entirely dependent on model implementation.
    model_dir = task_helper.paths.model_dir
    if task_helper.model_dir_is_empty():
        model = create_fake_model()
        logger.info('Model created from scratch')
    else:
        model = load_fake_model(model_dir)
        logger.info('Init model weights are loaded', extra={'model_dir': model_dir})

    # We will save weights of trained model (checkpoints) into directories provided by the checkpoints_saver.
    checkpoints_saver = task_helper.checkpoints_saver
    logger.info('Ready to save checkpoints', extra={'results_dir': task_helper.paths.results_dir})

    # Let's imitate reading input project with training data.
    # Of course in real implementations it is usually wrapped in some data loaders which are executed in parallel.
    project_meta = task_helper.in_project_meta  # Project meta contains list of project classes.
    project_dir = task_helper.paths.project_dir
    project_fs = sly.ProjectFS.from_disk_dir_project(project_dir)
    # ProjectFS enlists all samples (image/annotation pairs) in input project.
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
        logger.info('Read annotation from input project',
                    extra={'object_cnt': len(ann['objects']), 'tags': ann['tags']})

    # We are to report progress of task over sly.ProgressCounter if we want to observe the progress in web panel.
    # In fact one task may report progress for some sequential (not nested) subtasks,
    # but here we will report training progress only.
    progress = sly.progress_counter_train(cnt_epochs, cnt_iters_per_epoch)

    epoch_flt = 0
    for epoch in range(cnt_epochs):
        logger.info("Epoch started", extra={'epoch': epoch})

        for train_iter in range(cnt_iters_per_epoch):
            logger.info('Some forward-backward pass...')
            time.sleep(1)

            progress.iter_done_report()  # call it after every iteration to report progress
            epoch_flt = sly.epoch_float(epoch, train_iter + 1, cnt_iters_per_epoch)

            # And we are to report some metrics if we want to observe those amazing charts in web panel.
            # Regrettably, only the fixed metric types may be displayed now: 'loss', 'accuracy' and 'dice'.
            metric_values_train = {'loss': random.random(), 'my_metric': random.uniform(0, 100)}
            sly.report_metrics_training(epoch_flt, metric_values_train)

        logger.info("Epoch finished", extra={'epoch': epoch})

        # Validation is not necessary but may be performed. So let's imitate validation...
        logger.info("Validation...")
        time.sleep(2)
        # Metrics for validation may also be reported.
        metric_values_val = {'loss': random.random(), 'my_metric': random.uniform(0, 20)}
        sly.report_metrics_validation(epoch_flt, metric_values_val)

        # Save trained model weights when you want.
        # Model weights (checkpoint) should be written into directory provided by sly checkpoints_saver.
        # Content of the directory is entirely dependent on model implementation.
        cur_checkpoint_dir = checkpoints_saver.get_dir_to_write()
        dump_fake_model(cur_checkpoint_dir, model)
        checkpoints_saver.saved(is_best=True, optional_data={'epoch': epoch_flt, 'val_metrics': metric_values_val})
        # It is necessary to call checkpoints_saver.saved after saving.
        # By default new model will be created from the best checkpoint over the whole training
        # (which is determined by "is_best" flag).
        # Some optional info may be provided. It will be linked with the checkpoint
        # and may help to distinguish checkpoints from same training.

    # Thank you for your patience.
    logger.info('Training finished')


if __name__ == '__main__':
    main()
