import asyncio
import copy
import re
import sys
import weakref

from tqdm import tqdm

from supervisely.app.fastapi import run_sync
from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely.sly_logger import logger, EventType


def extract_by_regexp(regexp, string):
    result = re.search(regexp, string)
    if result is not None:
        return result.group(0)
    else:
        return ''


class _slyProgressBarIO:
    def __init__(self, widget_id, message=None, total=None):
        self.widget_id = widget_id

        self.progress = {
            'percent': 0,
            'info': '',
            'message': message,
            'status': None
        }

        self.prev_state = self.progress.copy()
        self.total = total
        self.tqdm_object = tqdm(disable=True)

    def print_progress_to_supervisely_tasks_section(self):
        '''
        Logs a message with level INFO on logger. Message contain type of progress, subtask message, currtnt and total number of iterations
        '''

        try:
            current_value = self.tqdm_object.n
        except ReferenceError:
            if self.total is not None:
                current_value = self.total
            else:
                return

        extra = {
            'event_type': EventType.PROGRESS,
            'subtask': self.progress.get('message', None),
            'current': current_value,
            'total': self.total,
        }

        gettrace = getattr(sys, "gettrace", None)
        in_debug_mode = gettrace is not None and gettrace()

        if not in_debug_mode:
            logger.info('progress', extra=extra)

    def write(self, s):
        new_text = s.strip().replace('\r', '')
        if len(new_text) != 0:
            if self.total is not None:
                self.progress['percent'] = self.tqdm_object.n / self.total * 100
                self.progress['info'] = extract_by_regexp(r'(\d+(?:\.\d+\w+)?)*/.*\]', new_text)
            else:
                self.progress['percent'] = self.tqdm_object.n
                self.progress['info'] = extract_by_regexp(r'(\d+(?:\.\d+\w+)?)*.*\]', new_text)

    def flush(self):
        if self.prev_state != self.progress:
            if self.progress['percent'] != '' and self.progress['info'] != '':
                self.print_progress_to_supervisely_tasks_section()

                for key, value in self.progress.items():
                    DataJson()[f'{self.widget_id}'][key] = value

                run_sync(DataJson().synchronize_changes())

                self.prev_state = copy.deepcopy(self.progress)

    def __del__(self):
        DataJson()[f'{self.widget_id}']['status'] = "success"
        DataJson()[f'{self.widget_id}']['percent'] = 100
        self.progress['percent'] = 100

        self.print_progress_to_supervisely_tasks_section()

        run_sync(DataJson().synchronize_changes())


class SlyTqdm(Widget):
    def __init__(self,
                 message: str = None,
                 show_percents: bool = False,
                 widget_id: str = None):
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

        super().__init__(widget_id=widget_id, file_path=__file__)

    def __call__(self, iterable=None, message=None, desc=None, total=None, leave=True, ncols=None,
                 mininterval=1.0, maxinterval=10.0, miniters=None, ascii=False, disable=False, unit='it',
                 unit_scale=False, dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0, position=None,
                 postfix=None, unit_divisor=1000, gui=False, **kwargs):

        class TqdmWithDestructor(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.sly_io = kwargs['file']

            def __del__(self):
                self.sly_io.write(self.__str__())
                self.sly_io.flush()

                del self.sly_io
                del self.fp
                self.disable = True

        extracted_total = copy.copy(tqdm(iterable=iterable, total=total, disable=True).total)

        sly_io = _slyProgressBarIO(self.widget_id, message, extracted_total)

        tqdm_object = TqdmWithDestructor(iterable=iterable, desc=desc, total=total, leave=leave, file=sly_io,
                                         ncols=ncols,
                                         mininterval=mininterval,
                                         maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable,
                                         unit=unit,
                                         unit_scale=unit_scale, dynamic_ncols=dynamic_ncols, smoothing=smoothing,
                                         bar_format=bar_format,
                                         initial=initial, position=position, postfix=postfix, unit_divisor=unit_divisor,
                                         gui=gui, **kwargs)

        sly_io.tqdm_object = weakref.proxy(tqdm_object)
        return tqdm_object

    def get_json_data(self):
        return {
            'percent': 0,
            'info': None,
            'message': self.message,
            'status': None,
            'show_percents': self.show_percents
        }

    def get_json_state(self):
        return None
