import numpy as np
from typing import Optional


class PredictionMask:
    def __init__(
        self, class_name: str, mask: np.ndarray, score: Optional[float] = None
    ):
        self.class_name = class_name
        self.mask = mask
        self.score = score
