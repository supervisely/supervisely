import asyncio
import copy
import re
import sys
import weakref

from tqdm import tqdm

from supervisely.app.fastapi import run_sync
from supervisely.app import DataJson
from supervisely.app.singleton import Singleton
from supervisely.app.widgets import Widget
from supervisely.sly_logger import logger, EventType


def extract_by_regexp(regexp, string):
    result = re.search(regexp, string)
    if result is not None:
        return result.group(0)
    else:
        return ""


class _slyProgressBarIO:
    def __init__(self, widget_id, message=None, total=None):
        self.widget_id = widget_id

        self.progress = {
            "percent": 0,
            "info": "",
            "message": message,
            "status": None,
            "n": -1,
        }

        self.prev_state = self.progress.copy()
        self.total = total

        self._n = 0

        self._done = False

    def print_progress_to_supervisely_tasks_section(self):
        """
        Logs a message with level INFO on logger. Message contain type of progress, subtask message, currtnt and total number of iterations
        """

        if self._n == self.progress["n"]:
            return

        self.progress["n"] = self._n
        extra = {
            "event_type": EventType.PROGRESS,
            "subtask": self.progress.get("message", None),
            "current": self._n,
            "total": self.total,
        }

        gettrace = getattr(sys, "gettrace", None)
        in_debug_mode = gettrace is not None and gettrace()

        if not in_debug_mode:
            logger.info("progress", extra=extra)

    def write(self, s):
        new_text = s.strip().replace("\r", "")
        if len(new_text) != 0:
            if self.total is not None:
                self.progress["percent"] = int(self._n / self.total * 100)
                self.progress["info"] = extract_by_regexp(
                    r"(\d+(?:\.\d+\w+)?)*/.*\]", new_text
                )
            else:
                self.progress["percent"] = int(self._n)
                self.progress["info"] = extract_by_regexp(
                    r"(\d+(?:\.\d+\w+)?)*.*\]", new_text
                )

    def flush(self, synchronize_changes=True):
        if self.prev_state != self.progress:

            if self.progress["percent"] != "" and self.progress["info"] != "":
                self.print_progress_to_supervisely_tasks_section()

                if self.progress["percent"] == 100 and self.total is not None:
                    self.progress["status"] = "success"

                for key, value in self.progress.items():
                    DataJson()[f"{self.widget_id}"][key] = value

                if synchronize_changes is True:
                    run_sync(DataJson().synchronize_changes())

                self.prev_state = copy.deepcopy(self.progress)

    def __del__(self):
        self.progress["status"] = "success"
        self.progress["percent"] = 100

        self.flush(synchronize_changes=False)
        self.print_progress_to_supervisely_tasks_section()


class CustomTqdm(tqdm):
    def __init__(self, widget_id, message, *args, **kwargs):

        extracted_total = copy.copy(
            tqdm(iterable=kwargs["iterable"], total=kwargs["total"], disable=True).total
        )

        super().__init__(
            file=_slyProgressBarIO(widget_id, message, extracted_total), *args, **kwargs
        )

    def refresh(self, *args, **kwargs):
        if self.fp is not None:
            self.fp._n = self.n
        super().refresh(*args, **kwargs)

    def close(self):
        self.refresh()
        super(CustomTqdm, self).close()
        if self.fp is not None:
            self.fp.__del__()

    def __del__(self):

        super(CustomTqdm, self).__del__()
        if self.fp is not None:
            self.fp.__del__()


class SlyTqdm(Widget):
    # @TODO: track all active sessions for one object and close them if new object inited
    def __init__(
        self, message: str = None, show_percents: bool = False, widget_id: str = None
    ):
        """
        Wrapper for classic tqdm progress bar.

            Parameters
            ----------
            identifier  : int, required
                HTML element identifier
            message  : int, optional
                Text message which displayed in HTML


            desc, total, leave, ncols, ... :
                Like in tqdm

        """
        self.message = message
        self.show_percents = show_percents

        self._active_session = None
        self._sly_io = None
        self._hide_on_finish = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _close_active_session(self):
        if self._active_session is not None:
            try:
                self._active_session.__del__()
                self._active_session = None
            except ReferenceError:
                pass

    def __call__(
        self,
        iterable=None,
        message=None,
        desc=None,
        total=None,
        leave=None,
        ncols=None,
        mininterval=1.0,
        maxinterval=10.0,
        miniters=None,
        ascii=False,
        disable=False,
        unit="it",
        unit_scale=False,
        dynamic_ncols=False,
        smoothing=0.3,
        bar_format=None,
        initial=0,
        position=None,
        postfix=None,
        unit_divisor=1000,
        gui=False,
        **kwargs,
    ):

        return CustomTqdm(
            widget_id=self.widget_id,
            iterable=iterable,
            desc=desc,
            total=total,
            leave=leave,
            message=message,
            ncols=ncols,
            mininterval=mininterval,
            maxinterval=maxinterval,
            miniters=miniters,
            ascii=ascii,
            disable=disable,
            unit=unit,
            unit_scale=unit_scale,
            dynamic_ncols=dynamic_ncols,
            smoothing=smoothing,
            bar_format=bar_format,
            initial=initial,
            position=position,
            postfix=postfix,
            unit_divisor=unit_divisor,
            gui=gui,
            **kwargs,
        )

    def get_json_data(self):
        return {
            "percent": 0,
            "info": None,
            "message": self.message,
            "status": None,
            "show_percents": self.show_percents,
        }

    def get_json_state(self):
        return None


class Progress(SlyTqdm):
    def __init__(
        self,
        message: str = None,
        show_percents: bool = False,
        hide_on_finish=True,
        widget_id: str = None,
    ):
        self.hide_on_finish = hide_on_finish
        super().__init__(
            message=message, show_percents=show_percents, widget_id=widget_id
        )

    pass
