from pathlib import Path
import cv2
import numpy as np
from .osnet import osnet_x1_0
from collections import OrderedDict
from supervisely import logger

try:
    # pylint: disable=import-error
    import torch
    from torch.nn import functional as F
except ImportError:
    logger.warning("torch is not installed, OSNet re-ID cannot be used.")


class OsnetReIDModel:
    def __init__(self, weights_path: Path = None, device: torch.device = torch.device("cpu"), half: bool = False):
        self.device = device
        self.half = half
        self.input_shape = (256, 128)
        if weights_path is None:
            self.model = osnet_x1_0(num_classes=1000, loss='softmax', pretrained=True, use_gpu=device)
        else:
            self.model = osnet_x1_0(num_classes=1000, loss='softmax', pretrained=False, use_gpu=device)
            self.load_pretrained_weights(weights_path)
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()

    def load_pretrained_weights(self, weight_path: Path):
        checkpoint = torch.load(weight_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_dict = self.model.state_dict()

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            key = k[7:] if k.startswith("module.") else k
            if key in model_dict and model_dict[key].size() == v.size():
                new_state_dict[key] = v

        model_dict.update(new_state_dict)
        self.model.load_state_dict(model_dict)

    def get_features(self, xyxys, img: np.ndarray):
        if xyxys.size == 0:
            return np.empty((0, 512))

        crops = self._get_crops(xyxys, img)
        with torch.no_grad():
            features = self.model(crops)
            features = F.normalize(features, dim=1).cpu().numpy()

        return features

    def _get_crops(self, xyxys, img):
        h, w = img.shape[:2]
        crops = []

        for box in xyxys:
            x1, y1, x2, y2 = box.round().astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop = cv2.resize(img[y1:y2, x1:x2], self.input_shape[::-1])
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            crop = torch.from_numpy(crop).permute(2, 0, 1)
            crops.append(crop)

        batch = torch.stack(crops).to(self.device, dtype=torch.float16 if self.half else torch.float32)
        return batch



class OsnetReIDInterface:
    def __init__(self, weights: Path, device: str = "cpu", fp16: bool = False):
        self.device = torch.device(device)
        self.fp16 = fp16
        self.model = OsnetReIDModel(weights, self.device, half=fp16)

    def inference(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        if detections is None or np.size(detections) == 0:
            return np.zeros((0, 512), dtype=np.float32)  # пустой набор фичей

        xyxys = detections[:, 0:4]  # left, top, right, bottom
        features = self.model.get_features(xyxys, image)
        return features

 
