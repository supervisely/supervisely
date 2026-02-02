# coding: utf-8
from __future__ import annotations

import inspect
import math
import re
from functools import partial, wraps
from typing import Dict, Optional, Union

from tqdm import tqdm

from supervisely._utils import is_development, is_production, sizeof_fmt
from supervisely.sly_logger import EventType, logger


# float progress of training, since zero
def epoch_float(epoch, train_it, train_its):
    return epoch + train_it / float(train_its)


class Progress:
    """
    Modules operations monitoring and displaying statistics of data processing. :class:`Progress<Progress>` object is immutable.

    :param message: Progress message e.g. "Images uploaded:", "Processing:".
    :type message: str
    :param total_cnt: Total count.
    :type total_cnt: int, optional
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

        address = 'https://app.supervisely.com/'
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
        total_cnt: Optional[int] = None,
        ext_logger: Optional[logger] = None,
        is_size: Optional[bool] = False,
        need_info_log: Optional[bool] = False,
        min_report_percent: Optional[int] = 1,
        log_extra: Optional[Dict[str, str]] = None,
        update_task_progress: Optional[bool] = True,
    ):
        self.is_size = is_size
        self.message = message
        self.total = total_cnt
        self.current = 0
        self.is_total_unknown = True if total_cnt in [None, 0] else False

        self.total_label = ""
        self.current_label = ""
        self._refresh_labels()

        self.reported_cnt = 0
        self.logger = logger if ext_logger is None else ext_logger
        self.report_every = max(1, math.ceil((total_cnt or 0) / 100 * min_report_percent))
        self.need_info_log = need_info_log
        self.log_extra = log_extra
        self.update_task_progress = update_task_progress

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
                sizeof_fmt(self.current) if self.is_total_unknown else sizeof_fmt(self.total)
            )
            self.current_label = sizeof_fmt(self.current)
        else:
            self.total_label = str(self.current if self.is_total_unknown else self.total)
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
            "total": (math.ceil(self.current) if self.is_total_unknown else math.ceil(self.total)),
        }

        if self.is_size:
            extra["current_label"] = self.current_label
            extra["total_label"] = self.total_label

        if self.log_extra:
            extra.update(self.log_extra)

        if self.update_task_progress:
            self.logger.info("progress", extra=extra)
        if self.need_info_log is True:
            self.logger.info(
                f"{self.message} [{self.current_label} / {self.total_label}]", extra=self.log_extra
            )

    def need_report(self) -> bool:
        if (
            (self.is_total_unknown)
            or (self.current >= self.total)
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

    def set(self, current: int, total: Optional[int], report: Optional[bool] = True) -> None:
        """
        Sets counter current value and total value and logs a message depending on current number of iterations.

        :param current: Current count.
        :type current: int
        :param total: Total count.
        :type total: int, optional
        :param report: Defines whether to report to log or not.
        :type report: bool
        :return: None
        :rtype: :class:`NoneType`
        """
        self.total = total

        self.is_total_unknown = True if self.total in [None, 0] else False
        self.current = current
        self.reported_cnt = 0
        self.report_every = max(1, math.ceil((total or 0) / 100))
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
        "Verification finished.",
        extra={"output": output, "event_type": EventType.TASK_VERIFIED},
    )


def _report_metrics(m_type, epoch, metrics):
    logger.info(
        "metrics",
        extra={
            "event_type": EventType.METRICS,
            "type": m_type,
            "epoch": epoch,
            "metrics": metrics,
        },
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


class SlyWrapFile:
    def __init__(self) -> None:
        self._pattern = "\\r(.*?)\\:"

    def write(self, msg):
        match = re.search(self._pattern, msg)
        if match:
            msg = match.group(1) + "..."
        logger.info(msg)


class tqdm_sly(tqdm, Progress):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # for self._upload_monitor()
        self._iteration_value = 0
        self._iteration_number = 0
        self._iteration_locked = False
        self._total_monitor_size = 0

        # self.n = None
        # self._total = kwargs.get("total", kwargs.get("total_cnt"))

        for _tqdm, _progress in {
            "total": "total_cnt",
            "desc": "message",
            "unit": "is_size",
            "unit_scale": "is_size",
        }.items():
            if kwargs.get(_tqdm) is not None and kwargs.get(_progress) is not None:
                raise ValueError(
                    f"Ambiguity error: Please specify only one of arguments: '{_tqdm}' or '{_progress}'."
                )

        kwargs_tqdm = self._handle_kwargs_tqdm(args, kwargs.copy())
        kwargs_tqdm.setdefault("unit_divisor", 1024)

        if is_development():
            tqdm.__init__(
                self,
                *args,
                **kwargs_tqdm,
            )
            self.offset = 0  # to prevent overfilling of tqdm in console
        else:
            for k, v in {
                "disable": True,
                "delay": 0,  # sec init delay
                "mininterval": 3,  # sec between reports
                "miniters": 0,
                # "file": SlyWrapFile(),
            }.items():
                kwargs_tqdm.setdefault(k, v)

            desc = args[1] if len(args) > 1 else kwargs.get("desc", "Processing")
            logger.info("%s ...", desc)

            tqdm.__init__(
                self,
                *args,
                **kwargs_tqdm,
            )

            kwargs = self._handle_args_and_kwargs_prod(args, kwargs)
            Progress.__init__(
                self,
                **kwargs,
            )
            self.n = 0

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        if is_development():
            yield from super().__iter__()
        else:
            # Inlining instance variables as locals (speed optimisation)
            iterable = self.iterable
            # self.disable = True

            mininterval = self.mininterval
            last_print_t = self.last_print_t
            last_print_n = self.last_print_n
            min_start_t = self.start_t + self.delay
            n = self.n
            time = self._time

            try:
                for obj in iterable:
                    yield obj
                    # Update and possibly print the progressbar.
                    # Note: does not call self.update(1) for speed optimisation.
                    n += 1
                    # self.iter_done_report()

                    if n - last_print_n >= self.miniters:
                        cur_t = time()
                        dt = cur_t - last_print_t
                        if dt >= mininterval and cur_t >= min_start_t:
                            Progress.need_report(self)
                            Progress.iters_done_report(self, (n - last_print_n))

                            last_print_n = self.n  # self.last_print_n
                            last_print_t = cur_t  # self.last_print_t
            finally:
                self.n = n
                self.close()

    def update(self, count):
        if is_development():
            if self.total is not None:
                count = min(count, self.total - self.offset)
                self.offset += count

            tqdm.update(self, count)

            if self.n == self.total:
                self.close()
        else:
            Progress.iters_done_report(self, count)
            self.n += count

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        return self.update(
            *args,
            **kwargs,
        )

    def _progress_monitor(self, monitor):
        if is_development() and self.n >= self.total:
            self.refresh()
            self.close()

        if monitor.bytes_read == 8192:
            self._total_monitor_size += monitor.len

        if self._total_monitor_size > self.total:
            self.total = self._total_monitor_size

        if not self._iteration_locked:
            if is_development():
                super().update(self._iteration_value + monitor.bytes_read - self.n)
            else:
                self.set_current_value(self._iteration_value + monitor.bytes_read, report=False)

        if is_production() and self.need_report():
            self.report_progress()

        if monitor.bytes_read == monitor.len and not self._iteration_locked:
            self._iteration_value += monitor.len
            self._iteration_number += 1
            self._iteration_locked = True
            if is_development():
                self.refresh()

        if monitor.bytes_read < monitor.len:
            self._iteration_locked = False

    def get_partial(self):
        return partial(self._progress_monitor)

    def _handle_kwargs_tqdm(self, args, kwargs_tqdm):
        # pop and convert every possible (and relevant) kwarg from Progress
        if len(args) < 2:  # i.e. 'desc' not set as a positional argument
            if kwargs_tqdm.get("message") is not None:
                kwargs_tqdm.setdefault("desc", kwargs_tqdm["message"])
                kwargs_tqdm.pop("message")
            else:
                kwargs_tqdm.setdefault("desc", "Processing")
        if len(args) < 3:  # i.e. 'total' not set as a positional argument
            if kwargs_tqdm.get("total_cnt") is not None:
                kwargs_tqdm.setdefault("total", kwargs_tqdm["total_cnt"])
                kwargs_tqdm.pop("total_cnt")
        if len(args) < 12:  # i.e. 'unit' not set as a positional argument
            if kwargs_tqdm.pop("is_size", None) == True:
                kwargs_tqdm["unit"] = "B"
                kwargs_tqdm["unit_scale"] = True
        return kwargs_tqdm

    def _handle_args_and_kwargs_prod(self, args: tuple, kwargs: dict) -> Dict[str, str]:
        # pop and convert every possible (and relevant) kwarg from tqdm
        # mention that tqdm is a prior parent class
        if len(args) < 2:  # i.e. 'desc' not set as a positional argument
            if kwargs.get("desc") is not None:
                kwargs.setdefault("message", kwargs["desc"])
                kwargs.pop("desc")
            else:
                kwargs.setdefault("message", "Processing")
        else:
            kwargs.setdefault("message", args[1])  # args[1]==desc
        if len(args) < 3:  # i.e. 'total' not set as a positional argument
            if kwargs.get("total") is not None:
                kwargs.setdefault("total_cnt", kwargs["total"])
                kwargs.pop("total")
        else:
            kwargs.setdefault("total_cnt", args[2])  # args[2]==total
        if len(args) < 12:  # i.e. 'unit' not set as a positional argument
            if kwargs.get("unit") in [
                "",
                "B",
                "k",
                "M",
                "G",
                "T",
                "P",
                "E",
                "Z",
            ] and kwargs.pop("unit_scale", None):
                kwargs["is_size"] = True
                kwargs.pop("unit")
        else:
            if (
                args[11] in ["", "B", "k", "M", "G", "T", "P", "E", "Z"] and args[12] == True
            ):  # i.e. unit=="B" and unit_scale==True
                kwargs["is_size"] = True

        tqdm_init_params = inspect.signature(tqdm.__init__).parameters.keys()
        for keyword in tqdm_init_params:
            if keyword in kwargs:
                kwargs.pop(keyword)

        # see original tqdm.__init__ for logic behaviour
        iterable_is_not_none = False if kwargs.get("iterable") is None and len(args) == 0 else True
        if kwargs.get("total_cnt") is None and iterable_is_not_none is True:
            try:
                iterable = kwargs.get("iterable", args[0])
                kwargs["total_cnt"] = len(iterable)
            except (TypeError, AttributeError):
                kwargs["total_cnt"] = None
        if kwargs.get("total_cnt") == float("inf"):
            # Infinite iterations, behave same as unknown
            kwargs["total_cnt"] = None

        return kwargs

    @classmethod
    def from_original_tqdm(
        cls,
        orig_tqdm: tqdm,
        **kwargs,
    ):
        # iterable=None, desc=None, total=None, leave=True, file=None,
        #          ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
        #          ascii=None, disable=False, unit='it', unit_scale=False,
        #          dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0,
        #          position=None, postfix=None, unit_divisor=1000, write_bytes=False,
        #          lock_args=None, nrows=None, colour=None, delay=0, gui=False,

        vrs = vars(orig_tqdm).copy()
        sgn = inspect.signature(orig_tqdm.__init__)

        kw = {}

        non_idempotent_args = ["miniters", "position"]

        def _handle_ni_args(name, vrs):
            if name == "miniters":
                return None if vrs[name] == 0 else vrs[name]
            if name == "position":
                return None if vrs["pos"] == 0 else -vrs["pos"]

        for name, param in sgn.parameters.items():
            if name in kwargs:
                kw[name] = kwargs[name]
            elif name in non_idempotent_args:
                kw[name] = _handle_ni_args(name, vrs)
            else:
                if name == "kwargs":
                    pass
                else:
                    try:
                        kw[name] = vrs[name]
                    except KeyError:
                        kw[name] = param.default

        # legacy tqdm 'nested' in kwargs
        kw.update({k: v for k, v in kwargs.items() if k not in kw})
        return cls(**kw)


def handle_original_tqdm(func):
    # Deprecated
    @wraps(func)
    def wrapper_original_tqdm(*args, **kwargs):
        cb_name = (
            "progress_size_cb" if func.__qualname__ == "FileApi.upload_directory" else "progress_cb"
        )

        spc = inspect.getfullargspec(func)

        if cb_name not in spc.args:
            raise ValueError(
                f"The '{cb_name}' parameter was not found in the '{func.__qualname__}'"
            )
        else:  # Note: (args, kwargs) both in spc.args
            idx = spc.args.index(cb_name)
            try:
                progress_cb = args[idx]
            except IndexError:
                progress_cb = kwargs.get(cb_name)

            _progress_cb = progress_cb

            # Start progress bar setup
            if progress_cb is not None and isinstance(progress_cb, tqdm):
                if not type(progress_cb) == tqdm_sly:
                    progress_cb.clear()
                    _progress_cb = tqdm_sly.from_original_tqdm(progress_cb)

            new_args = list(args).copy()
            new_kwargs = kwargs.copy()
            try:
                new_args[idx] = _progress_cb
            except IndexError:
                new_kwargs[cb_name] = _progress_cb

        try:
            result = func(*new_args, **new_kwargs)
        except Exception as e:
            # Ensure progress bar gets closed in case of an exception
            if progress_cb is not None and isinstance(progress_cb, tqdm):
                if not type(progress_cb) == tqdm_sly:
                    progress_cb.close()
                _progress_cb.close()
            raise e

        # close progress bar
        if progress_cb is not None and isinstance(progress_cb, tqdm):
            if not type(progress_cb) == tqdm_sly:
                progress_cb.close()
            _progress_cb.close()

        return result

    return wrapper_original_tqdm
