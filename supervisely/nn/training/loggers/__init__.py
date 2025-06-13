from supervisely.nn.training.loggers.base_train_logger import BaseTrainLogger
from supervisely.nn.training.loggers.tensorboard_logger import (
    TensorboardLogger,
    tensorboard_installed,
)

train_logger = BaseTrainLogger()


def setup_train_logger(name="tensorboard_logger"):
    global train_logger
    if name == "tensorboard_logger":
        if tensorboard_installed:
            if not isinstance(train_logger, TensorboardLogger):
                train_logger = TensorboardLogger()
        else:
            raise ImportError("TensorboardX is not installed")
    elif name == "default_logger":
        if not isinstance(train_logger, BaseTrainLogger):
            train_logger = BaseTrainLogger()
    else:
        raise ValueError(f"Logger {name} is not supported")
