import subprocess
from typing import Callable

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboardX import SummaryWriter


class BaseTrainLogger:
    def __init__(self):
        self._on_train_started_callbacks = []
        self._on_train_finished_callbacks = []
        self._on_epoch_started_callbacks = []
        self._on_epoch_finished_callbacks = []
        self._on_step_callbacks = []

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
        for callback in self._on_epoch_finished_callbacks:
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

    def _update_step(self, logs):
        for callback in self._on_step_callbacks:
            callback(logs)

    def add_on_step_callback(self, callback: Callable):
        self._on_step_callbacks.append(callback)

    def _log(self, logs: dict, idx: int):
        raise NotImplementedError

    def log(self, logs: dict, idx: int):
        self._log(logs, idx)
        self._update_step(logs)


class TensorboardLogger(BaseTrainLogger):
    def __init__(self, log_dir):

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        super().__init__()

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def start_tensorboard(self):
        """Start TensorBoard server in a subprocess"""
        self._tb_process = subprocess.Popen(
            [
                "tensorboard",
                "--logdir",
                self.log_dir,
                "--host=localhost",
                "--port=8001",
                "--load_fast=false",
            ]
        )
        print(f"Started TensorBoard")

    def _log(self, logs: dict, idx: int):
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, idx)

    def log_step(self, logs: dict, idx: int):
        self._log(logs, idx)

    def log_epoch(self, logs: dict, idx: int):
        self._log(logs, idx)


train_logger = TensorboardLogger("logs")
