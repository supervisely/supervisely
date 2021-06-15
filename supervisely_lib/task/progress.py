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
    def __init__(self, message, total_cnt, ext_logger=None, is_size=False, need_info_log=False, min_report_percent=1):
        '''
        :param message: str
        :param total_cnt: int
        :param ext_logger: Logger class object
        '''
        self.is_size = is_size
        self.message = message
        self.total = total_cnt
        self.current = 0
        self.is_total_unknown = total_cnt == 0

        self.total_label = ''
        self.current_label = ''
        self._refresh_labels()

        self.reported_cnt = 0
        self.logger = logger if ext_logger is None else ext_logger
        self.report_every = max(1, math.ceil(total_cnt / 100 * min_report_percent))
        self.need_info_log = need_info_log

        mb5 = 5 * 1024 * 1024
        if self.is_size and self.is_total_unknown:
            self.report_every = mb5  # 5mb

        mb1 = 1 * 1024 * 1024
        if self.is_size and self.is_total_unknown is False and self.report_every < mb1:
            self.report_every = mb1  # 1mb

        if self.is_size and self.is_total_unknown is False and self.total > 40 * 1024 * 1024 and self.report_every < mb5:
            self.report_every = mb5

        self.report_progress()

    def _refresh_labels(self):
        if self.is_size:
            self.total_label = sizeof_fmt(self.total) if self.total > 0 else sizeof_fmt(self.current)
            self.current_label = sizeof_fmt(self.current)
        else:
            self.total_label = str(self.total if self.total > 0 else self.current)
            self.current_label = str(self.current)

    def iter_done(self):
        '''
        Increments the current iteration counter by 1
        '''
        self.current += 1
        if self.is_total_unknown:
            self.total = self.current
        self._refresh_labels()

    def iters_done(self, count):
        '''
        Increments the current iteration counter by given count
        :param count: int
        '''
        self.current += count
        if self.is_total_unknown:
            self.total = self.current
        self._refresh_labels()

    def report_progress(self):
        self.print_progress()
        self.reported_cnt += 1

    def print_progress(self):
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
            extra['current_label'] = self.current_label
            extra['total_label'] = self.total_label

        self.logger.info('progress', extra=extra)
        if self.need_info_log is True:
            self.logger.info(f"{self.message} [{self.current_label} / {self.total_label}]")

    def need_report(self):
        if (self.current == self.total) \
                or (self.current % self.report_every == 0) \
                or ((self.reported_cnt - 1) < (self.current // self.report_every)):
            return True
        return False

    def report_if_needed(self):
        '''
        The function determines whether the message should be logged depending on current number of iterations
        '''
        if self.need_report():
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

    def set_current_value(self, value, report=True):
        '''
        Increments the current iteration counter by this value minus the current value of the counter and logs a message depending on current number of iterations
        :param value: int
        '''
        if report is True:
            self.iters_done_report(value - self.current)
        else:
            self.iters_done(value - self.current)

    def set(self, current, total, report=True):
        self.total = total
        if self.total != 0:
            self.is_total_unknown = False
        self.current = current
        self.reported_cnt = 0
        self.report_every = max(1, math.ceil(total / 100))
        self._refresh_labels()
        if report is True:
            self.report_if_needed()


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
