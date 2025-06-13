from supervisely.nn.training.loggers.base_train_logger import BaseTrainLogger
tensorboard_installed = False
try:
    from tensorboardX import SummaryWriter
    tensorboard_installed = True
except ImportError:
    pass


class TensorboardLogger(BaseTrainLogger):
    def __init__(self, log_dir=None):
        if log_dir is None:
            self.log_dir = None
            self.writer = None
        else:
            self.log_dir = log_dir
            self.writer = SummaryWriter(log_dir)
        super().__init__()

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
    
    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.writer = None

    def log(self, logs: dict, idx: int):
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, idx)
        self.writer.flush()

    def log_step(self, logs: dict):
        self.log(logs, self.step_idx)

    def log_epoch(self, logs: dict):
        self.log(logs, self.epoch_idx)
