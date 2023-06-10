import numpy as np
import functools
from fastapi import Request, BackgroundTasks
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import supervisely as sly
from supervisely.annotation.label import Label
from supervisely.nn.prediction_dto import Prediction, PredictionBBox
from supervisely.nn.inference.tracking.tracker_interface import TrackerInterface
from supervisely.nn.inference import Inference
import supervisely.nn.inference.tracking.functional as F


class BBoxTracking(Inference):
    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None,
    ):
        super().__init__(
            model_dir,
            custom_inference_settings,
            sliding_window_mode=None,
            use_gui=False,
        )

        try:
            self.load_on_device(model_dir, "cuda")
        except RuntimeError:
            self.load_on_device(model_dir, "cpu")

    def get_info(self):
        info = super().get_info()
        info["task type"] = "tracking"
        return info

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/track")
        def start_track(request: Request, task: BackgroundTasks):
            task.add_task(track, request)
            return {"message": "Track task started."}

        def send_error_data(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                value = None
                try:
                    value = func(*args, **kwargs)
                except Exception as e:
                    request: Request = args[0]
                    context = request.state.context
                    api: sly.Api = request.state.api
                    track_id = context["trackId"]

                    api.post(
                        "videos.notify-annotation-tool",
                        data={
                            "type": "videos:tracking-error",
                            "data": {
                                "trackId": track_id,
                                "error": {"message": repr(e)},
                            },
                        },
                    )
                return value

            return wrapper

        @send_error_data
        def track(request: Request):
            context = request.state.context
            api: sly.Api = request.state.api

            video_interface = TrackerInterface(
                context=context,
                api=api,
                load_all_frames=False,
            )
            api.logger.info("Start tracking.")

            for _ in video_interface.frames_loader_generator():
                for (fig_id, geom), obj_id in zip(
                    video_interface.geometries.items(),
                    video_interface.object_ids,
                ):
                    if isinstance(geom, sly.Rectangle):
                        imgs = video_interface.frames_with_notification
                        target = PredictionBBox("", [geom.top, geom.left, geom.bottom, geom.right])
                        geometry = self.predict(
                            rgb_image=imgs[-1],
                            init_rgb_image=imgs[0],
                            target_bbox=target,
                            settings=self.custom_inference_settings_dict,
                        )
                    else:
                        raise TypeError(f"Tracking does not work with {geom.geometry_name()}.")

                    video_interface.add_object_geometries([geometry], obj_id, fig_id)
                    api.logger.info(f"Object #{obj_id} tracked.")

                    if video_interface.global_stop_indicatior:
                        return

    def predict(
        self,
        rgb_image: np.ndarray,
        init_rgb_image: np.ndarray,
        target_bbox: PredictionBBox,
        settings: Dict[str, Any],
    ) -> PredictionBBox:
        """
        SOT prediction

        :param rgb_image: image for searching
        :type rgb_image: np.ndarray
        :param init_rgb_image: first frame with object
        :type init_rgb_image: np.ndarray
        :param target_bbox: initial annotation
        :type target_bbox: PredictionBBox
        :param settings: model parameters
        :type settings: Dict[str, Any]
        :return: predicted annotation
        :rtype: PredictionBBox
        """
        raise NotImplementedError

    def visualize(
        self,
        predictions: List[PredictionBBox],
        images: List[np.ndarray],
        vis_path: str,
        thickness: int = 2,
    ):
        vis_path = Path(vis_path)

        for i, (pred, image) in enumerate(zip(predictions, images)):
            out_path = vis_path / f"img_{i}.jpg"
            ann = self._predictions_to_annotation(image, [pred])
            ann.draw_pretty(
                bitmap=image,
                color=(255, 0, 0),
                thickness=thickness,
                output_path=str(out_path),
                fill_rectangles=False,
            )

    def _create_label(self, dto: PredictionBBox) -> sly.Rectangle:
        top, left, bottom, right = dto.bbox_tlbr
        geometry = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
        return Label(geometry, sly.ObjClass("", sly.Rectangle))

    def _get_obj_class_shape(self):
        return sly.Rectangle

    def _predictions_to_annotation(
        self, image: np.ndarray, predictions: List[Prediction]
    ) -> sly.Annotation:
        labels = []
        for prediction in predictions:
            label = self._create_label(prediction)
            if label is None:
                # for example empty mask
                continue
            if isinstance(label, list):
                labels.extend(label)
                continue
            labels.append(label)

        # create annotation with correct image resolution
        ann = sly.Annotation(img_size=image.shape[:2])
        ann = ann.add_labels(labels)
        return ann
