# coding: utf-8

import math

from supervisely_lib.sly_logger import logger, EventType
from supervisely_lib._utils import sizeof_fmt

# float progress of training, since zero
def epoch_float(epoch, train_it, train_its):
    return epoch + train_it / float(train_its)


class Progress:
    '''
    This is a class for conveniently monitoring the operation of modules and displaying statistics on data processing
    '''
    def __init__(self, message, total_cnt, ext_logger=None, is_size=False, need_info_log=False):
        '''
        :param message: str
        :param total_cnt: int
        :param ext_logger: Logger class object
        '''
        self.is_size = is_size
        self.message = message
        self.total = total_cnt
        self.current = 0
        self.reported_cnt = 0
        self.logger = logger if ext_logger is None else ext_logger
        self.report_every = max(1, math.ceil(total_cnt / 100))
        self.need_info_log = need_info_log
        self.report_progress()

    def iter_done(self):
        '''
        Increments the current iteration counter by 1
        '''
        self.current += 1

    def iters_done(self, count):
        '''
        Increments the current iteration counter by given count
        :param count: int
        '''
        self.current += count

    #@TODO: ask web team to rename subtask->message
    def report_progress(self):
        '''
        Logs a message with level INFO on logger. Message contain type of progress, subtask message, currtnt and total number of iterations
        '''
        extra = {
            'event_type': EventType.PROGRESS,
            'subtask': self.message,
            'current': math.ceil(self.current),
            'total': math.ceil(self.total) if self.total > 0 else math.ceil(self.current),
        }

        if self.is_size:
            extra['current_label'] = sizeof_fmt(self.current)
            extra['total_label'] = sizeof_fmt(self.total) if self.total > 0 else sizeof_fmt(self.current)

        self.logger.info('progress', extra=extra)
        self.reported_cnt += 1

        if self.need_info_log is True:
            self.logger.info(f"{self.message} [{extra['current']} / {extra['total']}]")

    def report_if_needed(self):
        '''
        The function determines whether the message should be logged depending on current number of iterations
        '''
        if (self.current == self.total) \
                or (self.current % self.report_every == 0) \
                or ((self.reported_cnt - 1) < (self.current // self.report_every)):
            self.report_progress()

    def iter_done_report(self):  # finish & report
        '''
        Increments the current iteration counter by 1 and logs a message depending on current number of iterations
        :return:
        '''
        self.iter_done()
        self.report_if_needed()

    def iters_done_report(self, count):  # finish & report
        '''
        Increments the current iteration counter by given count and logs a message depending on current number of iterations
        :param count: int
        '''
        self.iters_done(count)
        self.report_if_needed()

    #@TODO: to upload dtl archive
    def set_current_value(self, value):
        '''
        Increments the current iteration counter by this value minus the current value of the counter and logs a message depending on current number of iterations
        :param value: int
        '''
        self.iters_done_report(value - self.current)


def report_agent_rpc_ready():
    '''
    Logs a message with level INFO on logger
    '''
    logger.info('Ready to get events', extra={ 'event_type': EventType.TASK_DEPLOYED })


def report_import_finished():
    '''
    Logs a message with level INFO on logger
    '''
    logger.info('import finished', extra={'event_type': EventType.IMPORT_APPLIED})


def report_inference_finished():
    '''
    Logs a message with level INFO on logger
    '''
    logger.info('model applied', extra={'event_type': EventType.MODEL_APPLIED})


def report_dtl_finished():
    '''
    Logs a message with level INFO on logger
    '''
    logger.info('DTL finished', extra={'event_type': EventType.DTL_APPLIED})


def report_dtl_verification_finished(output):
    '''
    Logs a message with level INFO on logger
    :param output: str
    '''
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
