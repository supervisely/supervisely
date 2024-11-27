from time import time
from typing import List, Union
from PIL import Image
from PIL.Image import Image as PIL_Image
import numpy as np
from supervisely import ProjectMeta


class SlyModel():
    def __init__(self):
        self.model_meta: ProjectMeta = None
    
    def load_model(
        self,
        model_files: dict,
        device: str = "cuda",
        runtime: str = "PyTorch",
        load_options: dict = None,
    ):
        pass
    
    def predict(self, image: np.ndarray, settings: dict = None):
        pass

    def predict_batch(self, images: List[np.ndarray], settings: dict = None):
        pass
    
    def predict_benchmark(self, images: List[np.ndarray], settings: dict = None):
        pass
    
    def inference(self, input: Union[str, np.ndarray, PIL_Image, list], settings: dict = None):
        if not isinstance(input, list):
            single_input = True
            input = [input]
        images = [self.read_input(img) for img in input]
        predictions = self._inference_auto(images, settings)
        if single_input:
            return predictions[0]
        else:
            return predictions
    
    @staticmethod
    def read_input(input: Union[str, np.ndarray, PIL_Image]):
        if isinstance(input, np.ndarray):
            return input
        elif isinstance(input, str):
            return np.asarray(Image.open(input).convert("RGB"))
        elif isinstance(input, PIL_Image):
            return np.asarray(input)
        else:
            raise ValueError("Unsupported input type.")
    
    def _inference_auto(
        self,
        images: List[np.ndarray],
        settings: dict = None,
    ):
        if self.is_batch_inference_supported():
            if self._is_predict_batch_implemented():
                return self.predict_batch(images, settings)
            else:
                return self.predict_benchmark(images, settings)[0]
        else:
            return [self.predict(img, settings) for img in images]

    def _inference_benchmark(
        self,
        images: List[np.ndarray],
        settings: dict = None,
    ):
        t0 = time()
        if self._is_predict_benchmark_implemented():
            predictions, benchmark = self.predict_benchmark(images, settings)
        elif self._is_predict_batch_implemented():
            predictions = self.predict_batch(images, settings)
            benchmark = {}
        elif len(images) == 1:
            predictions = [self.predict(images[0], settings)]
            benchmark = {}
        else:
            raise NotImplementedError("Batch inference is not implemented.")
        total_time = (time() - t0) * 1000  # milliseconds
        benchmark = {
            "total": total_time,
            "preprocess": benchmark.get("preprocess"),
            "inference": benchmark.get("inference"),
            "postprocess": benchmark.get("postprocess"),
        }
        return predictions, benchmark
    
    def is_batch_inference_supported(self):
        return self._is_predict_batch_implemented() or self._is_predict_benchmark_implemented()

    def _is_predict_batch_implemented(self):
        return type(self).predict_batch != SlyModel.predict_batch
    
    def _is_predict_benchmark_implemented(self):
        return type(self).predict_benchmark != SlyModel.predict_benchmark