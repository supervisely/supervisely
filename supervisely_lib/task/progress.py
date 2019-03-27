# coding: utf-8

import math

from supervisely_lib.sly_logger import logger, EventType


# float progress of training, since zero
def epoch_float(epoch, train_it, train_its):
    return epoch + train_it / float(train_its)


class Progress:
    def __init__(self, message, total_cnt, ext_logger=None):
        self.message = message
        self.total = total_cnt
        self.current = 0
        self.reported_cnt = 0
        self.logger = logger if ext_logger is None else ext_logger
        self.report_every = max(1, math.ceil(total_cnt / 100))
        self.report_progress()

    def iter_done(self):
        self.current += 1

    def iters_done(self, count):
        self.current += count

    #@TODO: ask web team to rename subtask->message
    def report_progress(self):
        self.logger.info('progress', extra={
            'event_type': EventType.PROGRESS,
            'subtask': self.message,
            'current': math.ceil(self.current),
            'total': math.ceil(self.total),
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

    #@TODO: to upload dtl archive
    def set_current_value(self, value):
        self.iters_done_report(value - self.current)


def report_agent_rpc_ready():
    logger.info('Ready to get events', extra={ 'event_type': EventType.TASK_DEPLOYED })


def report_import_finished():
    logger.info('import finished', extra={'event_type': EventType.IMPORT_APPLIED})


def report_inference_finished():
    logger.info('model applied', extra={'event_type': EventType.MODEL_APPLIED})


def report_dtl_finished():
   logger.info('DTL finished', extra={'event_type': EventType.DTL_APPLIED})


def report_dtl_verification_finished(output):
   logger.info('Verification finished.', extra={'output': output, 'event_type': EventType.TASK_VERIFIED})


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


def report_checkpoint_saved(checkpoint_idx, subdir, sizeb, best_now, optional_data):
    logger.info('checkpoint', extra={
        'event_type': EventType.CHECKPOINT,
        'id': checkpoint_idx,
        'subdir': subdir,
        'sizeb': sizeb,
        'best_now': best_now,
        'optional': optional_data
    })
