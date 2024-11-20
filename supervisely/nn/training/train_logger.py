import os
import socket
import subprocess
from typing import Callable

from tensorboard import program
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboardX import SummaryWriter

from supervisely._utils import is_production


class BaseTrainLogger:
    def __init__(self):
        self._on_train_started_callbacks = []
        self._on_train_finished_callbacks = []
        self._on_epoch_started_callbacks = []
        self._on_epoch_finished_callbacks = []
        self._on_step_callbacks = []
        self.epoch_idx = 0
        self.step_idx = 0

    def train_started(self, total_epochs: int):
        for callback in self._on_train_started_callbacks:
            callback(total_epochs)

    def train_finished(self):
        for callback in self._on_train_finished_callbacks:
            callback()

    def epoch_started(self, total_steps: int):
        for callback in self._on_epoch_started_callbacks:
            callback(total_steps)

    def epoch_finished(self):
        self.epoch_idx += 1
        for callback in self._on_epoch_finished_callbacks:
            callback()

    def on_step_end(self):
        self.step_idx += 1
        for callback in self._on_step_callbacks:
            callback()

    # def step_started(self):
    # for callback in self._on_step_callbacks:
    # callback()

    def add_on_train_started_callback(self, callback: Callable):
        self._on_train_started_callbacks.append(callback)

    def add_on_train_finish_callback(self, callback: Callable):
        self._on_train_finished_callbacks.append(callback)

    def add_on_epoch_started_callback(self, callback: Callable):
        self._on_epoch_started_callbacks.append(callback)

    def add_on_epoch_finish_callback(self, callback: Callable):
        self._on_epoch_finished_callbacks.append(callback)

    def add_on_step_callback(self, callback: Callable):
        self._on_step_callbacks.append(callback)

    def _log(self, logs: dict, idx: int):
        raise NotImplementedError

    def _log_step(self, logs: dict):
        raise NotImplementedError

    def _log_epoch(self, logs: dict):
        raise NotImplementedError

    def log(self, logs: dict, idx: int):
        self._log(logs, idx)

    def log_step(self, logs: dict):
        self._log_step(logs)

    def log_epoch(self, logs: dict):
        self._log_epoch(logs)


class TensorboardLogger(BaseTrainLogger):
    def __init__(self, log_dir):

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.tensorboard_process = None
        super().__init__()

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def start_tensorboard(self):
        def get_host_ip():
            try:
                host_ip = socket.gethostbyname(socket.gethostname())
                return host_ip
            except Exception as e:
                print(f"Error retrieving host IP: {e}")
                return "localhost"

        # self.tensorboard_process = subprocess.Popen(
        #     [
        #         "tensorboard",
        #         "--logdir",
        #         self.log_dir,
        #         "--host=localhost",
        #         "--port=8001",
        #         "--load_fast=false",
        #     ]
        # )

        if is_production():
            host = "0.0.0.0"
        else:
            host = "localhost"

        self.tensorboard_process = program.TensorBoard()
        self.tensorboard_process.configure(
            argv=[
                None,
                "--logdir",
                self.log_dir,
                "--host",
                host,
                "--port",
                "8001",
                "--load_fast",
                "false",
            ]
        )

        url = self.tensorboard_process.launch()
        if is_production():
            url = f"http://{get_host_ip()}:8001"

        print(f"TensorBoard is running at {url}")
        return url

    def stop_tensorboard(self):
        """Stop Tensorboard server"""
        # if self.tensorboard_process is not None:
        #     self.tensorboard_process.terminate()
        pid = self.tensorboard_process._server._process.pid  # Get the process ID
        os.kill(pid, 9)  # Send a termination signal
        print("TensorBoard has been stopped.")

    def _log(self, logs: dict, idx: int):
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, idx)
        self.writer.flush()

    def _log_step(self, logs: dict):
        self._log(logs, self.step_idx)

    def _log_epoch(self, logs: dict):
        self._log(logs, self.epoch_idx)


train_logger = TensorboardLogger("logs")
