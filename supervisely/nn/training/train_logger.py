import subprocess
from typing import Callable

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboardX import SummaryWriter


class BaseTrainLogger:
    def __init__(self):
        self._on_train_started_callbacks = []
        self._on_epoch_started_callbacks = []
        self._on_step_callbacks = []

    def train_started(self, total_epochs: int):
        for callback in self._on_train_started_callbacks:
            callback(total_epochs)

    def epoch_started(self, total_steps: int):
        for callback in self._on_epoch_started_callbacks:
            callback(total_steps)

    def _update_step(self, logs):
        for callback in self._on_step_callbacks:
            callback(logs)

    def add_on_train_started_callback(self, callback: Callable):
        self._on_train_started_callbacks.append(callback)

    def add_on_epoch_started_callback(self, callback: Callable):
        self._on_epoch_started_callbacks.append(callback)

    def add_on_step_callback(self, callback: Callable):
        self._on_step_callbacks.append(callback)

    def _log(self, logs):
        raise NotImplementedError

    def log(self, logs):
        self._log(logs)
        self._update_step(logs)


class TensorboardLogger(BaseTrainLogger):
    def __init__(self, log_dir):

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.idx = 0
        super().__init__()

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

    def _log(self, logs: dict):
        self.idx += 1
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, self.idx)


train_logger = TensorboardLogger("logs")
