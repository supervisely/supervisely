import subprocess
from time import sleep
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
        """Start Tensorboard server in a subprocess"""
        args = [
            "tensorboard",
            "--logdir",
            self.log_dir,
            "--host=localhost",
            "--port=8001",
            "--load_fast=false",
            "--reload_multifile=true",
        ]
        self.tensorboard_process = subprocess.Popen(args)
        print(f"Tensorboard server has been started")

    def stop_tensorboard(self):
        """Stop Tensorboard server"""
        if self.tensorboard_process is not None:
            self.tensorboard_process.terminate()
            print(f"Tensorboard server has been stopped")
        else:
            print("Tensorboard server is not running")

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
