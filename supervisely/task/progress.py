# coding: utf-8
from __future__ import annotations

import math
from typing import Optional
from supervisely.sly_logger import logger, EventType
from supervisely._utils import sizeof_fmt


# float progress of training, since zero
def epoch_float(epoch, train_it, train_its):
    return epoch + train_it / float(train_its)


class Progress:
    """
    Modules operations monitoring and displaying statistics of data processing. :class:`Progress<Progress>` object is immutable.

    :param message: Progress message e.g. "Images uploaded:", "Processing:".
    :type message: str
    :param total_cnt: Total count.
    :type total_cnt: int
    :param ext_logger: Logger object.
    :type ext_logger: logger, optional
    :param is_size: Shows Label size.
    :type is_size: bool, optional
    :param need_info_log: Shows info log.
    :type need_info_log: bool, optional
    :param min_report_percent: Minimum report percent of total items in progress to log.
    :type min_report_percent: int, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.sly_logger import logger

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        progress = sly.Progress("Images downloaded: ", len(img_infos), ext_logger=logger, is_size=True, need_info_log=True)
        api.image.download_paths(ds_id, image_ids, save_paths, progress_cb=progress.iters_done_report)

        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 0,
        #  "total": 6, "current_label": "0.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:45.659Z", "level": "info"}
        # {"message": "Images downloaded:  [0.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:45.660Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 1,
        #  "total": 6, "current_label": "1.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.134Z", "level": "info"}
        # {"message": "Images downloaded:  [1.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.134Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 2,
        #  "total": 6, "current_label": "2.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "Images downloaded:  [2.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 3,
        #  "total": 6, "current_label": "3.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "Images downloaded:  [3.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 4,
        #  "total": 6, "current_label": "4.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "Images downloaded:  [4.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 5,
        #  "total": 6, "current_label": "5.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.136Z", "level": "info"}
        # {"message": "Images downloaded:  [5.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.136Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 6,
        #  "total": 6, "current_label": "6.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.136Z", "level": "info"}
        # {"message": "Images downloaded:  [6.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.136Z", "level": "info"}
    """

    def __init__(
        self,
        message: str,
        total_cnt: int,
        ext_logger: Optional[logger] = None,
        is_size: Optional[bool] = False,
        need_info_log: Optional[bool] = False,
        min_report_percent: Optional[int] = 1,
    ):
        self.is_size = is_size
        self.message = message
        self.total = total_cnt
        self.current = 0
        self.is_total_unknown = total_cnt == 0

        self.total_label = ""
        self.current_label = ""
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

        if (
            self.is_size
            and self.is_total_unknown is False
            and self.total > 40 * 1024 * 1024
            and self.report_every < mb5
        ):
            self.report_every = mb5

        self.report_progress()

    def _refresh_labels(self):
        if self.is_size:
            self.total_label = (
                sizeof_fmt(self.total) if self.total > 0 else sizeof_fmt(self.current)
            )
            self.current_label = sizeof_fmt(self.current)
        else:
            self.total_label = str(self.total if self.total > 0 else self.current)
            self.current_label = str(self.current)

    def iter_done(self) -> None:
        """
        Increments the current iteration counter by 1
        """
        self.current += 1
        if self.is_total_unknown:
            self.total = self.current
        self._refresh_labels()

    def iters_done(self, count: int) -> None:
        """
        Increments the current iteration counter by given count

        :param count: Amount of iters
        :type count: int
        """
        self.current += count
        if self.is_total_unknown:
            self.total = self.current
        self._refresh_labels()

    def report_progress(self) -> None:
        """
        Logs a message with level INFO in logger. Message contain type of progress, subtask message, current and total number of iterations

        :return: None
        :rtype: :class:`NoneType`
        """
        self.print_progress()
        self.reported_cnt += 1

    def print_progress(self) -> None:
        """
        Logs a message with level INFO on logger. Message contain type of progress, subtask message, currtnt and total number of iterations
        """
        extra = {
            "event_type": EventType.PROGRESS,
            "subtask": self.message,
            "current": math.ceil(self.current),
            "total": math.ceil(self.total) if self.total > 0 else math.ceil(self.current),
        }

        if self.is_size:
            extra["current_label"] = self.current_label
            extra["total_label"] = self.total_label

        self.logger.info("progress", extra=extra)
        if self.need_info_log is True:
            self.logger.info(f"{self.message} [{self.current_label} / {self.total_label}]")

    def need_report(self) -> bool:
        if (
            (self.current >= self.total)
            or (self.current % self.report_every == 0)
            or ((self.reported_cnt - 1) < (self.current // self.report_every))
        ):
            return True
        return False

    def report_if_needed(self) -> None:
        """
        Determines whether the message should be logged depending on current number of iterations
        """
        if self.need_report():
            self.report_progress()

    def iter_done_report(self) -> None:  # finish & report
        """
        Increments the current iteration counter by 1 and logs a message depending on current number of iterations.

        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            progress = sly.Progress("Processing:", len(img_infos))
            for img_info in img_infos:
                img_names.append(img_info.name)
                progress.iter_done_report()

            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 0, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 1, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 2, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 3, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 4, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 5, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 6, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
        """
        self.iter_done()
        self.report_if_needed()

    def iters_done_report(self, count: int) -> None:  # finish & report
        """
        Increments the current iteration counter by given count and logs a message depending on current number of iterations.

        :param count: Counter.
        :type count: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            progress = sly.Progress("Processing:", len(img_infos))
            for img_info in img_infos:
                img_names.append(img_info.name)
                progress.iters_done_report(1)

            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 0, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 1, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 2, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 3, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 4, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 5, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 6, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
        """
        self.iters_done(count)
        self.report_if_needed()

    def set_current_value(self, value: int, report: Optional[bool] = True) -> None:
        """
        Increments the current iteration counter by this value minus the current value of the counter and logs a message depending on current number of iterations.

        :param value: Current value.
        :type value: int
        :param report: Defines whether to report to log or not.
        :type report: bool
        :return: None
        :rtype: :class:`NoneType`
        """
        if report is True:
            self.iters_done_report(value - self.current)
        else:
            self.iters_done(value - self.current)

    def set(self, current: int, total: int, report: Optional[bool] = True) -> None:
        """
        Sets counter current value and total value and logs a message depending on current number of iterations.

        :param current: Current count.
        :type current: int
        :param total: Total count.
        :type total: int
        :param report: Defines whether to report to log or not.
        :type report: bool
        :return: None
        :rtype: :class:`NoneType`
        """
        self.total = total
        if self.total != 0:
            self.is_total_unknown = False
        self.current = current
        self.reported_cnt = 0
        self.report_every = max(1, math.ceil(total / 100))
        self._refresh_labels()
        if report is True:
            self.report_if_needed()


def report_agent_rpc_ready() -> None:
    """
    Logs a message with level INFO on logger
    """
    logger.info("Ready to get events", extra={"event_type": EventType.TASK_DEPLOYED})


def report_import_finished() -> None:
    """
    Logs a message with level INFO on logger
    """
    logger.info("import finished", extra={"event_type": EventType.IMPORT_APPLIED})


def report_inference_finished() -> None:
    """
    Logs a message with level INFO on logger
    """
    logger.info("model applied", extra={"event_type": EventType.MODEL_APPLIED})


def report_dtl_finished() -> None:
    """
    Logs a message with level INFO on logger
    """
    logger.info("DTL finished", extra={"event_type": EventType.DTL_APPLIED})


def report_dtl_verification_finished(output: str) -> None:
    """
    Logs a message with level INFO on logger
    :param output: str
    """
    logger.info(
        "Verification finished.", extra={"output": output, "event_type": EventType.TASK_VERIFIED}
    )


def _report_metrics(m_type, epoch, metrics):
    logger.info(
        "metrics",
        extra={"event_type": EventType.METRICS, "type": m_type, "epoch": epoch, "metrics": metrics},
    )


def report_metrics_training(epoch, metrics):
    _report_metrics("train", epoch, metrics)


def report_metrics_validation(epoch, metrics):
    _report_metrics("val", epoch, metrics)


def report_checkpoint_saved(checkpoint_idx, subdir, sizeb, best_now, optional_data) -> None:
    logger.info(
        "checkpoint",
        extra={
            "event_type": EventType.CHECKPOINT,
            "id": checkpoint_idx,
            "subdir": subdir,
            "sizeb": sizeb,
            "best_now": best_now,
            "optional": optional_data,
        },
    )
