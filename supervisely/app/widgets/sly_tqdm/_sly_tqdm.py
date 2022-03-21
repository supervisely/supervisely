import asyncio
import copy
import re

from asgiref.sync import async_to_sync
from tqdm import tqdm

from supervisely.app import DataJson
from supervisely.app.widgets import Widget


def extract_by_regexp(regexp, string):
    result = re.search(regexp, string)
    if result is not None:
        return result.group(0)
    else:
        return ''


class _slyProgressBarIO:
    def __init__(self, widget_id, message=None, total_provided=None):
        self.widget_id = widget_id

        self.progress = {
            'percent': 0,
            'info': '',
            'message': message,
            'status': None
        }

        self.prev_state = self.progress.copy()
        self.total_provided = total_provided

    def write(self, s):
        print(s)
        new_text = s.strip().replace('\r', '')
        if len(new_text) != 0:
            if self.total_provided:
                self.progress['percent'] = extract_by_regexp(r'\d*\%', new_text).replace('%', '')
                self.progress['info'] = extract_by_regexp(r'(\d+(?:\.\d+\w+)?)*/.*\]', new_text)
            else:
                self.progress['percent'] = 50
                self.progress['info'] = extract_by_regexp(r'(\d+(?:\.\d+\w+)?)*.*\]', new_text)

    def flush(self):
        if self.prev_state != self.progress:
            if self.progress['percent'] != '0' and self.progress['percent'] != '' and self.progress['info'] != '':

                for key, value in self.progress.items():
                    DataJson()[f'{self.widget_id}'][key] = value

                DataJson().synchronize_changes()

                self.prev_state = copy.deepcopy(self.progress)

    def __del__(self):
        DataJson()[f'{self.widget_id}']['status'] = "success"
        DataJson()[f'{self.widget_id}']['percent'] = 100

        DataJson().synchronize_changes()


class sly_tqdm(Widget):
    def __init__(self,
                 message: str = None,
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

        super().__init__(widget_id=widget_id, file_path=__file__)

    def __call__(self, iterable=None, message=None, desc=None, total=None, leave=True, ncols=None,
                 mininterval=1.0, maxinterval=10.0, miniters=None, ascii=False, disable=False, unit='it',
                 unit_scale=False, dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0, position=None,
                 postfix=None, unit_divisor=1000, gui=False, **kwargs):

        total_provided = True if (iterable is not None and hasattr(iterable, '__len__') and len(
            iterable) is not None) or total is not None else False

        sly_io = _slyProgressBarIO(self.widget_id, message, total_provided)

        if total_provided is not None:
            return tqdm(iterable=iterable, desc=desc, total=total, leave=leave, file=sly_io, ncols=ncols,
                        mininterval=mininterval,
                        maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable, unit=unit,
                        unit_scale=unit_scale, dynamic_ncols=dynamic_ncols, smoothing=smoothing, bar_format=bar_format,
                        initial=initial, position=position, postfix=postfix, unit_divisor=unit_divisor,
                        gui=gui, **kwargs)

    def get_serialized_data(self):
        return {
            'percent': 0,
            'info': None,
            'message': self.message,
            'status': None
        }

    def get_serialized_state(self):
        return None
