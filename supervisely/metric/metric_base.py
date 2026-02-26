# coding: utf-8


class MetricsBase:
    """Abstract base class for metrics that compare ground-truth and prediction annotations."""

    def add_pair(self, ann_gt, ann_pred):
        raise NotImplementedError()

    def get_metrics(self):
        raise NotImplementedError()

    def get_total_metrics(self):
        raise NotImplementedError()

    def log_total_metrics(self):
        raise NotImplementedError()
