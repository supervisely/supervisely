import subprocess

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboardX import SummaryWriter

from supervisely.nn.training.loggers.base_train_logger import BaseTrainLogger


class TensorboardLogger(BaseTrainLogger):
    def __init__(self, log_dir=None):
        if log_dir is None:
            self.log_dir = None
            self.writer = None
        else:
            self.log_dir = log_dir
            self.writer = SummaryWriter(log_dir)
        self.tensorboard_process = None
        super().__init__()

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def start_tensorboard(self):
        """Start Tensorboard server in a subprocess"""
        if self.log_dir is None:
            self.log_dir = "logs"

        args = [
            "tensorboard",
            "--logdir",
            self.log_dir,
            "--host=localhost",
            "--port=8001",
            "--load_fast=true",
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


tb_logger = TensorboardLogger()
