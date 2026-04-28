# coding: utf-8
"""
Single prediction returned by deployed model.
"""

from __future__ import annotations

import atexit
import os
import tempfile
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import requests

from supervisely._utils import get_valid_kwargs, logger, rand_str
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagValueType
from supervisely.api.api import Api
from supervisely.convert.image.sly.sly_image_helper import get_meta_from_annotation
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging.image import read as read_image
from supervisely.imaging.image import read_bytes as read_image_bytes
from supervisely.imaging.image import write as write_image
from supervisely.io.fs import (
    clean_dir,
    dir_empty,
    ensure_base_path,
    get_file_ext,
    mkdir,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.video.video import VideoFrameReader


class Prediction:
    """
    A single prediction result.

    The prediction is primarily represented by ``annotation_json`` (Supervisely annotation format).
    If you provide ``model_meta`` (a :class:`~supervisely.project.project_meta.ProjectMeta` or its JSON),
    you can access the parsed :class:`~supervisely.annotation.annotation.Annotation` via
    :attr:`~supervisely.nn.model.prediction.Prediction.annotation`.
    """
    _temp_dir = os.path.join(tempfile.gettempdir(), "prediction_files")
    __cleanup_registered = False

    def __init__(
        self,
        annotation_json: Dict,
        source: Union[str, int] = None,
        model_meta: Optional[Union[ProjectMeta, Dict]] = None,
        name: Optional[str] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
        project_id: Optional[int] = None,
        dataset_id: Optional[int] = None,
        image_id: Optional[int] = None,
        video_id: Optional[int] = None,
        frame_index: Optional[int] = None,
        api: Optional["Api"] = None,
        **kwargs,
    ):
        """
        :param annotation_json: Prediction in Supervisely annotation JSON format.
        :type annotation_json: dict or :class:`~supervisely.annotation.annotation.Annotation`
        :param source: Optional source descriptor (path, id, etc.), used as a hint for loading/visualization.
        :type source: str or int, optional
        :param model_meta: Output meta of the model, required to construct an :class:`~supervisely.annotation.annotation.Annotation`.
        :type model_meta: :class:`~supervisely.project.project_meta.ProjectMeta` or dict, optional
        :param name: Optional name for this prediction (used in visualization filenames).
        :type name: str, optional
        :param path: Local path to the source image/video.
        :type path: str, optional
        :param url: URL to the source image/video.
        :type url: str, optional
        :param project_id: Project id associated with the source.
        :type project_id: int, optional
        :param dataset_id: Dataset id associated with the source.
        :type dataset_id: int, optional
        :param image_id: Image id associated with the source.
        :type image_id: int, optional
        :param video_id: Video id associated with the source (for frame predictions).
        :type video_id: int, optional
        :param frame_index: Frame index for video predictions.
        :type frame_index: int, optional
        :param api: API client used to download image/frame by id.
        :type api: :class:`~supervisely.api.api.Api`, optional
        """
        self.source = source
        if isinstance(annotation_json, Annotation):
            annotation_json = annotation_json.to_json()

        self.annotation_json = annotation_json
        self.model_meta = model_meta
        if isinstance(self.model_meta, dict):
            self.model_meta = ProjectMeta.from_json(self.model_meta)

        self.name = name
        self.path = path
        self.url = url
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.image_id = image_id
        self.video_id = video_id
        self.frame_index = frame_index
        self.extra_data = {}
        if kwargs:
            self.extra_data.update(kwargs)
        self.api = api

        self._annotation = None
        self._boxes = None
        self._masks = None
        self._classes = None
        self._scores = None
        self._track_ids = None

        if self.path is None and isinstance(self.source, (str, PathLike)):
            self.path = str(self.source)

    def _init_geometries(self):
        """Initialize the geometries of the prediction."""
        def _get_confidence(label: Label):
            for tag_name in ["confidence", "conf", "score"]:
                conf_tag: Tag = label.tags.get(tag_name, None)
                if conf_tag is not None and conf_tag.meta.value_type != str(
                    TagValueType.ANY_NUMBER
                ):
                    conf_tag = None
                if conf_tag is not None:
                    return conf_tag.value
            return 1

        self._boxes = []
        self._masks = []
        self._classes = []
        self._scores = []
        for label in self.annotation.labels:
            self._classes.append(label.obj_class.name)
            self._scores.append(_get_confidence(label))
            if isinstance(label.geometry, Rectangle):
                rect = label.geometry
            else:
                rect = label.geometry.to_bbox()
            if isinstance(label.geometry, Bitmap):
                self._masks.append(label.geometry.get_mask(self.annotation.img_size))

            self._boxes.append(
                np.array(
                    [
                        rect.top,
                        rect.left,
                        rect.bottom,
                        rect.right,
                    ]
                )
            )
        self._boxes = np.array(self._boxes)
        self._masks = np.array(self._masks)

        custom_data = self.annotation.custom_data
        if custom_data and isinstance(custom_data, list) and len(custom_data) == len(self.annotation.labels):
            self._track_ids = np.array(custom_data)

    @property
    def boxes(self):
        """Get the bounding boxes of the prediction."""
        if self._boxes is None:
            self._init_geometries()
        return self._boxes

    @property
    def masks(self):
        """Get the masks of the prediction."""
        if self._masks is None:
            self._init_geometries()
        return self._masks

    @property
    def classes(self):
        """Get the classes of the prediction."""
        if self._classes is None:
            self._init_geometries()
        return self._classes

    @property
    def scores(self):
        """Get the scores of the prediction."""
        if self._scores is None:
            self._init_geometries()
        return self._scores

    @property
    def annotation(self) -> Annotation:
        """Get the annotation of the prediction."""
        if self._annotation is None and self.annotation_json is not None:
            if self.model_meta is None:
                raise ValueError("Model meta is not provided. Cannot create annotation.")
            model_meta = get_meta_from_annotation(self.annotation_json, self.model_meta)
            self._annotation = Annotation.from_json(self.annotation_json, model_meta)
        return self._annotation

    @annotation.setter
    def annotation(self, annotation: Union[Annotation, Dict]):
        """Set the annotation of the prediction."""
        if isinstance(annotation, Annotation):
            self._annotation = annotation
            self.annotation_json = annotation.to_json()
        elif isinstance(annotation, dict):
            self._annotation = None
            self.annotation_json = annotation
        else:
            raise ValueError("Annotation must be either a dict or an Annotation object.")

    @property
    def class_idxs(self) -> np.ndarray:
        """Get the class indexes of the prediction."""
        if self.model_meta is None:
            raise ValueError("Model meta is not provided. Cannot create class indexes.")
        cls_name_to_idx = {
            obj_class.name: i for i, obj_class in enumerate(self.model_meta.obj_classes)
        }
        return np.array([cls_name_to_idx[class_name] for class_name in self.classes])
    @property
    def track_ids(self):
        """Get track IDs for each detection. Returns None for detections without tracking.

        :returns: Track IDs for each detection.
        :rtype: np.ndarray
        """
        if self._track_ids is None:
            self._init_geometries()
        return self._track_ids

    @classmethod
    def from_json(cls, json_data: Dict, **kwargs) -> "Prediction":
        """
        Create :class:`~supervisely.nn.model.prediction.Prediction` from a JSON dict.

        This helper accepts both styles of keys coming from different backends:
        ``annotation_json`` or ``annotation``.

        :param json_data: Source JSON.
        :type json_data: dict
        :param kwargs: Additional fields to override/extend JSON (e.g. ``api``, ``model_meta``).
        :type kwargs: dict
        :returns: Prediction instance.
        :rtype: :class:`~supervisely.nn.model.prediction.Prediction`
        :raises ValueError: If annotation payload is missing.
        """

        kwargs = {**json_data, **kwargs}
        if "annotation_json" in kwargs:
            annotation_json = kwargs.pop("annotation_json")
        elif "annotation" in kwargs:
            annotation_json = kwargs.pop("annotation")
        else:
            raise ValueError("Annotation JSON is required.")
        kwargs = get_valid_kwargs(
            kwargs,
            Prediction.__init__,
            exclude=["self", "annotation_json"],
        )
        return cls(annotation_json, **kwargs)

    def to_json(self):
        """
        Serialize prediction to a JSON-compatible dict.

        :returns: JSON dict.
        :rtype: dict
        """

        return {
            "source": self.source,
            "annotation": self.annotation_json,
            "name": self.name,
            "path": self.path,
            "url": self.url,
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "image_id": self.image_id,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            **self.extra_data,
        }

    def _clear_temp_files(self):
        if not dir_empty(self._temp_dir):
            clean_dir(self._temp_dir)

    def load_image(self) -> np.ndarray:
        """
        Load the source image (or video frame) into a NumPy array.

        The loader uses the first available source in the following order:
        ``path`` → ``url`` → ``image_id`` (requires ``api`` or environment config) →
        ``video_id`` + ``frame_index`` → video ``path``/``url`` + ``frame_index``.

        :returns: Image (or frame) as ``H×W×C`` NumPy array.
        :rtype: numpy.ndarray
        """

        api = self.api
        if self.frame_index is None:
            if self.path is not None:
                return read_image(self.path)
            if self.url is not None:
                ext = get_file_ext(self.url)
                if ext == "":
                    ext = ".jpg"
                r = requests.get(self.url, stream=True, timeout=60)
                r.raise_for_status()
                return read_image_bytes(r.content)
            if self.image_id is not None:
                try:
                    if api is None:
                        # TODO: raise more clarifying error in case of failing of api init
                        # what a user should do to fix it?
                        api = Api()
                    return api.image.download_np(self.image_id)
                except Exception as e:
                    raise RuntimeError("Failed to load image by ID") from e
            raise ValueError("Cannot load image. No path, URL, image ID, or frame_index provided.")
        if self.video_id is not None:
            if self.frame_index is None:
                raise ValueError("Frame index is not provided for video.")
            try:
                if api is None:
                    api = Api()
                return api.video.frame.download_np(self.video_id, self.frame_index)
            except Exception as e:
                raise RuntimeError("Failed to load frame by video ID") from e
        if self.path is not None:
            return next(VideoFrameReader(self.path, frame_indexes=[self.frame_index]))
        if self.url is not None:
            video_name = Path(self.url).name
            video_path = Path(self._temp_dir) / video_name
            mkdir(self._temp_dir)
            if not video_path.exists():
                with requests.get(self.url, stream=True, timeout=10 * 60) as r:
                    r.raise_for_status()
                    with open(video_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            return next(VideoFrameReader(video_path, frame_indexes=[self.frame_index]))
        raise ValueError("Cannot load frame. No path, URL or video ID provided.")

    def visualize(
        self,
        save_path: Optional[str] = None,
        save_dir: Optional[str] = None,
        color: Optional[List[int]] = None,
        thickness: Optional[int] = None,
        opacity: Optional[float] = 0.5,
        draw_tags: Optional[bool] = False,
        fill_rectangles: Optional[bool] = True,
    ) -> np.ndarray:
        """
        Render prediction on top of the source image/frame.

        If ``save_path`` (file or directory) or ``save_dir`` is provided, the visualization is saved.
        Otherwise the rendered image is returned.

        :param save_path: Output file path or directory.
        :type save_path: str, optional
        :param save_dir: Output directory (alias for directory ``save_path``).
        :type save_dir: str, optional
        :param color: Optional RGB color for drawing.
        :type color: List[int], optional
        :param thickness: Optional line thickness.
        :type thickness: int, optional
        :param opacity: Bitmap opacity.
        :type opacity: float, optional
        :param draw_tags: If True, draws tags on objects.
        :type draw_tags: bool, optional
        :param fill_rectangles: If True, fills rectangles.
        :type fill_rectangles: bool, optional
        :returns: Rendered image.
        :rtype: numpy.ndarray
        """

        if save_dir is not None and save_path is not None:
            raise ValueError("Only one of save_path or save_dir can be provided.")

        mkdir(self._temp_dir)
        if not Prediction.__cleanup_registered:
            atexit.register(self._clear_temp_files)
            Prediction.__cleanup_registered = True

        img = self.load_image()
        self.annotation.draw_pretty(
            bitmap=img,
            color=color,
            thickness=thickness,
            opacity=opacity,
            draw_tags=draw_tags,
            output_path=None,
            fill_rectangles=fill_rectangles,
        )
        if save_dir is not None:
            save_path = save_dir
        if save_path is None:
            return img
        if Path(save_path).suffix == "":
            # is a directory
            if self.name is not None:
                name = self.name
                if self.frame_index is not None:
                    name = f"{name}_{self.frame_index}"
            elif self.image_id is not None:
                name = str(self.image_id)
            elif self.video_id is not None:
                name = str(self.video_id)
            elif self.path is not None:
                name = Path(self.path).name
                if self.frame_index is not None:
                    name = f"{name}_{self.frame_index}"
            else:
                name = f"{rand_str(6)}"
            name = f"vis_{name}.png"
            save_path = os.path.join(save_path, name)
        ensure_base_path(save_path)
        write_image(save_path, img)
        logger.info("Visualization for prediction saved to %s", save_path)
        return img
