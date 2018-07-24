# coding: utf-8

import math

from ..sly_logger import logger, EventType


# float progress of training, since zero
def epoch_float(epoch, train_it, train_its):
    res = epoch + train_it / float(train_its)
    return res


class ProgressCounter:
    def __init__(self, subtask_name, total_cnt,
                 report_limit=100, ext_logger=None, report_divisor=1):
        self.subtask = subtask_name
        self.total = total_cnt
        self.current = 0
        self.report_limit = report_limit
        self.report_divisor = report_divisor
        self.reported_cnt = 0
        if ext_logger is None:
            self.logger = logger
        else:
            self.logger = ext_logger

        if self.report_limit < 1:
            self.report_every = 1
        else:
            self.report_every = max(1, int(math.ceil(total_cnt / report_limit)))

        self.report_progress()

    def iter_done(self):
        self.current += 1

    def iters_done(self, count):
        self.current += count

    def report_progress(self):
        self.logger.info('progress', extra={
            'event_type': EventType.PROGRESS,
            'subtask': self.subtask,
            'current': math.ceil(self.current / self.report_divisor),
            'total': math.ceil(self.total / self.report_divisor),
        })
        self.reported_cnt += 1

    def report_if_needed(self):
        if (self.current == self.total) \
                or (self.current % self.report_every == 0) \
                or ((self.reported_cnt - 1) < (self.current // self.report_every)):
            self.report_progress()

    def iter_done_report(self):  # finish & report
        self.iter_done()
        self.report_if_needed()

    def iters_done_report(self, count):  # finish & report
        self.iters_done(count)
        self.report_if_needed()


def progress_counter_train(total_epochs, cnt_iters_per_epoch, ext_logger=None):
    total_cnt = total_epochs * cnt_iters_per_epoch
    ctr = ProgressCounter('model training', total_cnt, ext_logger=ext_logger)
    return ctr


def progress_counter_dtl(pr_name, cnt_images, ext_logger=None):
    total_cnt = cnt_images
    ctr = ProgressCounter('DTL: {}'.format(pr_name), total_cnt, ext_logger=ext_logger)
    return ctr


def progress_counter_inference(cnt_imgs, ext_logger=None):
    ctr = ProgressCounter('model applying', cnt_imgs, ext_logger=ext_logger)
    return ctr


def progress_download_project(total_cnt, pr_name, ext_logger=None):
    subtask_name = 'Downloading: {}'.format(pr_name)
    ctr = ProgressCounter(subtask_name=subtask_name, total_cnt=total_cnt, ext_logger=ext_logger)
    return ctr


def progress_upload_project(total_cnt, pr_name, ext_logger=None):
    subtask_name = 'Uploading: {}'.format(pr_name)
    ctr = ProgressCounter(subtask_name=subtask_name, total_cnt=total_cnt, ext_logger=ext_logger)
    return ctr


def progress_download_nn(total_cnt, ext_logger=None):
    subtask_name = 'Download NN, MB:'
    ctr = ProgressCounter(subtask_name=subtask_name, total_cnt=total_cnt, ext_logger=ext_logger,
                          report_divisor=2**20)  # by MBs
    return ctr


def progress_counter_import(pr_name, cnt_images, ext_logger=None):
    total_cnt = cnt_images
    ctr = ProgressCounter('Import: {}'.format(pr_name), total_cnt, ext_logger=ext_logger)
    return ctr


# metrics as dct {string_name: float_value}
def _report_metrics(m_type, epoch, metrics):
    logger.info('metrics', extra={
        'event_type': EventType.METRICS,
        'type': m_type,
        'epoch': epoch,
        'metrics': metrics
    })


def report_metrics_training(epoch, metrics):
    _report_metrics('train', epoch, metrics)


def report_metrics_validation(epoch, metrics):
    _report_metrics('val', epoch, metrics)


def report_inference_finished():
    logger.info('model applied', extra={
        'event_type': EventType.MODEL_APPLIED,
    })


def report_import_finished():
    logger.info('import finished', extra={
        'event_type': EventType.IMPORT_APPLIED,
    })


def report_agent_rpc_ready():
    logger.info('Ready to get events', extra={
        'event_type': EventType.TASK_DEPLOYED,
    })
