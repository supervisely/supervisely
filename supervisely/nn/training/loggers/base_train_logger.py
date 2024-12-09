from typing import Callable


class BaseTrainLogger:
    def __init__(self):
        self._on_train_started_callbacks = []
        self._on_train_finished_callbacks = []
        self._on_epoch_started_callbacks = []
        self._on_epoch_finished_callbacks = []
        self._on_step_finished_callbacks = []
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

    def step_finished(self):
        self.step_idx += 1
        for callback in self._on_step_finished_callbacks:
            callback()

    def add_on_train_started_callback(self, callback: Callable):
        self._on_train_started_callbacks.append(callback)

    def add_on_train_finish_callback(self, callback: Callable):
        self._on_train_finished_callbacks.append(callback)

    def add_on_epoch_started_callback(self, callback: Callable):
        self._on_epoch_started_callbacks.append(callback)

    def add_on_epoch_finish_callback(self, callback: Callable):
        self._on_epoch_finished_callbacks.append(callback)

    def add_on_step_finished_callback(self, callback: Callable):
        self._on_step_finished_callbacks.append(callback)

    def log(self, logs: dict, idx: int):
        raise NotImplementedError

    def log_step(self, logs: dict):
        raise NotImplementedError

    def log_epoch(self, logs: dict):
        raise NotImplementedError
    
    def close(self):
        pass
