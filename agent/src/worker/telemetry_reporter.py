# coding: utf-8

import time
import traceback
import subprocess

from hurry.filesize import size as bytes_to_human
import supervisely_lib as sly
from supervisely_lib import EventType
import supervisely_lib.worker_proto as api_proto

from .task_logged import TaskLogged
from . import constants
from .system_info import get_directory_size_bytes


class TelemetryReporter(TaskLogged):
    @classmethod
    def _get_subprocess_out(cls, in_str):
        res = subprocess.Popen([in_str],
                               shell=True, executable="/bin/bash",
                               stdout=subprocess.PIPE).communicate()[0]
        return res

    def _get_subprocess_out_if_possible(self, proc_id, in_str):
        no_output = b''
        if proc_id in self.skip_subproc:
            return no_output
        res = self._get_subprocess_out(in_str)
        if len(res) <= 2:  # cr lf
            self.skip_subproc.add(proc_id)
            return no_output
        return res

    def __init__(self):
        super().__init__({'task_id': 'telemetry'})
        self.skip_subproc = set()

    def init_logger(self):
        super().init_logger()
        sly.change_formatters_default_values(self.logger, 'worker', 'telemetry')

    def task_main_func(self):
        try:
            self.logger.info('TELEMETRY_REPORTER_INITIALIZED')

            def data_stream_gen():
                while True:
                    time.sleep(1)
                    htop_str = 'echo q | htop -C | ' \
                               'aha --line-fix | html2text -width 999 | grep -v "F1Help" | grep -v "xml version="'
                    htop_output = self._get_subprocess_out_if_possible('htop', htop_str)

                    nvsmi_str = 'echo q | nvidia-smi | aha --line-fix | html2text -width 999 | grep -v "xml version="'
                    nvsmi_output = self._get_subprocess_out_if_possible('nvsmi', nvsmi_str)

                    img_sizeb = get_directory_size_bytes(self.data_mgr.storage.images.storage_root_path)
                    nn_sizeb = get_directory_size_bytes(self.data_mgr.storage.nns.storage_root_path)
                    tasks_sizeb = get_directory_size_bytes(constants.AGENT_TASKS_DIR)
                    node_storage = [
                        {'Images': bytes_to_human(img_sizeb)},
                        {'NN weights': bytes_to_human(nn_sizeb)},
                        {'Tasks': bytes_to_human(tasks_sizeb)},
                        {'Total': bytes_to_human(img_sizeb + nn_sizeb + tasks_sizeb)},
                    ]

                    server_info = {
                        'htop': htop_output.decode("utf-8"),
                        'nvsmi': nvsmi_output.decode("utf-8"),
                        'node_storage': node_storage
                    }

                    info_str = sly.json_dumps(server_info)
                    yield api_proto.AgentInfo(info=info_str)

            self.api.put_endless_stream('UpdateTelemetry', api_proto.Empty, data_stream_gen())

        except Exception:
            extra = {'event_type': EventType.TASK_CRASHED, 'error': traceback.format_exc().split('\n')}
            self.logger.critical("TELEMETRY_REPORTER_CRASHED", extra={**extra})
