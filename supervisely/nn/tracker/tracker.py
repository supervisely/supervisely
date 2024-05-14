import argparse
import os
from typing import Dict, List, Union

import numpy as np

import supervisely as sly
from supervisely import Annotation
from supervisely.nn.tracker.ann_keeper import AnnotationKeeper
from supervisely.sly_logger import logger


class BaseDetection:
    def get_sly_label(self):
        raise NotImplementedError()


class BaseTrack:
    def __init__(self, track_id, *args, **kwargs):
        self.track_id = track_id

    def get_sly_label(self):
        raise NotImplementedError()

    def clean_sly_label(self):
        raise NotImplementedError()


class BaseTracker:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.args = self.parse_settings(settings)
        self.device = self.select_device(device=self.args.device)

    def select_device(self, device="", batch_size=None):
        import torch

        # device = 'cpu' or '0' or '0,1,2,3'
        cpu_request = device.lower() == "cpu"
        if device and not cpu_request:  # if device requested other than 'cpu'
            os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
            assert (
                torch.cuda.is_available()
            ), f"CUDA unavailable, invalid device {device} requested"  # check availablity

        cuda = False if cpu_request else torch.cuda.is_available()
        if cuda:
            c = 1024**2  # bytes to MB
            ng = torch.cuda.device_count()
            if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
                assert (
                    batch_size % ng == 0
                ), f"batch-size {batch_size} not multiple of GPU count {ng}"
            x = [torch.cuda.get_device_properties(i) for i in range(ng)]
            s = f"Using torch {torch.__version__} "
            for i, d in enumerate((device or "0").split(",")):
                if i == 1:
                    s = " " * len(s)
                logger.info(f"{s}CUDA:{d} ({x[i].name}, {x[i].total_memory / c}MB)")
        else:
            logger.info(f"Using torch {torch.__version__} CPU")

        logger.info("")  # skip a line
        return torch.device("cuda:0" if cuda else "cpu")

    def parse_settings(self, settings: Dict) -> argparse.Namespace:
        _settings = self.default_settings()
        _settings.update(settings)
        if "device" not in _settings:
            _settings["device"] = ""
        return argparse.Namespace(**_settings)

    def default_settings(self):
        """To be overridden by subclasses."""
        return {}

    def track(
        self,
        images: Union[List[np.ndarray], List[str]],
        frame_to_annotation: Dict[int, Annotation],
        ann_keeper: AnnotationKeeper,
        pbar_cb=None,
    ):
        raise NotImplementedError()

    def convert_annotation(self, annotation_for_frame: sly.Annotation):
        formatted_predictions = []
        sly_labels = []

        for label in annotation_for_frame.labels:
            confidence = 1.0
            if label.tags.get("confidence", None) is not None:
                confidence = label.tags.get("confidence").value
            elif label.tags.get("conf", None) is not None:
                confidence = label.tags.get("conf").value

            rectangle: sly.Rectangle = label.geometry.to_bbox()
            formatted_pred = [
                rectangle.left,
                rectangle.top,
                rectangle.right,
                rectangle.bottom,
                confidence,
            ]

            # convert to width / height
            formatted_pred[2] -= formatted_pred[0]
            formatted_pred[3] -= formatted_pred[1]

            formatted_predictions.append(formatted_pred)
            sly_labels.append(label)

        return formatted_predictions, sly_labels

    def correct_figure(self, img_size, figure):  # img_size — height, width tuple
        # check figure is within image bounds
        canvas_rect = sly.Rectangle.from_size(img_size)
        if canvas_rect.contains(figure.to_bbox()) is False:
            # crop figure
            figures_after_crop = [cropped_figure for cropped_figure in figure.crop(canvas_rect)]
            if len(figures_after_crop) > 0:
                return figures_after_crop[0]
            else:
                return None
        else:
            return figure

    def update_track_data(
        self, tracks_data: dict, tracks: List[BaseTrack], frame_index: int, img_size
    ):
        coordinates_data = []
        track_id_data = []
        labels_data = []

        for curr_track in tracks:
            track_id = curr_track.track_id - 1  # tracks in deepsort started from 1

            if curr_track.get_sly_label() is not None:
                track_id_data.append(track_id)
                labels_data.append(curr_track.get_sly_label())

        tracks_data[frame_index] = {"ids": track_id_data, "labels": labels_data}

        return tracks_data

    def clear_empty_ids(self, tracker_annotations):
        id_mappings = {}
        last_ordinal_id = 0

        for frame_index, data in tracker_annotations.items():
            data_ids_temp = []
            for current_id in data["ids"]:
                new_id = id_mappings.get(current_id, -1)
                if new_id == -1:
                    id_mappings[current_id] = last_ordinal_id
                    last_ordinal_id += 1
                    new_id = id_mappings.get(current_id, -1)
                data_ids_temp.append(new_id)
            data["ids"] = data_ids_temp

        return tracker_annotations
