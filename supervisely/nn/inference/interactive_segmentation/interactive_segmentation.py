import os
from fastapi import Response, Request, status

from supervisely.geometry.bitmap import Bitmap
from supervisely.nn.prediction_dto import PredictionSegmentation
from supervisely.sly_logger import logger
from supervisely.imaging import image as sly_image
from supervisely.io.fs import silent_remove
from supervisely._utils import rand_str
from supervisely.app.content import get_data_dir
from supervisely.nn.inference import Inference
from supervisely import ProjectMeta, ObjClass, Label
from supervisely.nn.inference.interactive_segmentation import functional

from typing import Dict, List, Any, Optional, Union

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class InteractiveSegmentation(Inference):
    class Click:
        def __init__(self, x, y, is_positive):
            self.x = x
            self.y = y
            self.is_positive = is_positive

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}: {str(self.__dict__)}"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None,
        sliding_window_mode: Optional[Literal["basic", "advanced", "none"]] = "basic",
        use_gui: Optional[bool] = False,
    ):
        super().__init__(model_dir, custom_inference_settings, sliding_window_mode, use_gui)
        self._current_smtool_state = None
        self._current_image_path = None
        self._class_names = ["mask_prediction"]
        color = [255, 0, 0]
        self._model_meta = ProjectMeta([ObjClass(self._class_names[0], Bitmap, color)])

    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "interactive segmentation"
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        info["tracking_on_videos_support"] = False
        return info

    def _get_obj_class_shape(self):
        return Bitmap

    def _create_label(self, dto: PredictionSegmentation):
        classes = self.get_classes()
        assert len(classes) == 1, "InteractiveSegmentation can't be used for multi-class inference"
        obj_class = self.model_meta.get_obj_class(classes[0])
        if not dto.mask.any():  # skip empty masks
            logger.debug(f"Mask of class {dto.class_name} is empty and will be skipped")
            return None
        geometry = Bitmap(dto.mask)
        label = Label(geometry, obj_class)
        return label

    def predict(
        self,
        image_path: str,
        clicks: List[Click],
        settings: Dict[str, Any],
    ) -> PredictionSegmentation:
        raise NotImplementedError("Have to be implemented in child class")

    def get_classes(self) -> List[str]:
        return self._class_names

    def _check_image_changed(self, smtool_state):
        if self._current_smtool_state is not None:
            prev_state = self._current_smtool_state.copy()
            smtool_state = smtool_state.copy()
            smtool_state.pop("positive")
            smtool_state.pop("negative")
            prev_state.pop("positive")
            prev_state.pop("negative")
            return smtool_state != prev_state
        else:
            return True

    def _on_model_deployed(self):
        self._reset_current_state()

    def _reset_current_state(self):
        self._current_smtool_state = None
        self._current_image_path = None

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/smart_segmentation")
        def smart_segmentation(response: Response, request: Request):
            # 1. parse request
            # 2. download image
            # 3. make crop
            # 4. predict

            logger.debug(
                f"smart_segmentation inference: context=",
                extra={**request.state.context, "api_token": "***"},
            )

            try:
                state = request.state.state
                settings = self._get_inference_settings(state)
                smtool_state = request.state.context
                api = request.state.api
                crop = smtool_state["crop"]
                positive_clicks, negative_clicks = (
                    smtool_state["positive"],
                    smtool_state["negative"],
                )
                if len(positive_clicks) + len(negative_clicks) == 0:
                    logger.warn("No clicks received.")
                    response = {
                        "origin": None,
                        "bitmap": None,
                        "success": True,
                        "error": None,
                    }
                    return response
            except Exception as exc:
                logger.warn("Error parsing request:" + str(exc), exc_info=True)
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "400: Bad request.", "success": False}

            # if something has changed (except clicks) we will re-download the image
            is_image_changed = self._check_image_changed(smtool_state)
            if is_image_changed:
                if self._current_image_path is not None:
                    silent_remove(self._current_image_path)
                self._reset_current_state()
                app_dir = get_data_dir()
                image_np = functional.download_image_from_context(smtool_state, api, app_dir)
                image_np = functional.crop_image(crop, image_np)
                self._current_image_path = os.path.join(app_dir, f"{rand_str(10)}.jpg")
                sly_image.write(self._current_image_path, image_np)

            self._current_smtool_state = smtool_state

            clicks = [{**click, "is_positive": True} for click in positive_clicks]
            clicks += [{**click, "is_positive": False} for click in negative_clicks]
            clicks = functional.transform_clicks_to_crop(crop, clicks)
            is_in_bbox = functional.validate_click_bounds(crop, clicks)
            if not is_in_bbox:
                logger.warn(f"Invalid value: click is out of bbox bounds.")
                return {
                    "origin": None,
                    "bitmap": None,
                    "success": True,
                    "error": None,
                }

            # predict
            clicks_to_predict = [self.Click(c["x"], c["y"], c["is_positive"]) for c in clicks]
            pred_mask = self.predict(self._current_image_path, clicks_to_predict, settings)
            pred_mask = pred_mask.mask

            if pred_mask.any():
                bitmap = Bitmap(pred_mask)
                bitmap_origin, bitmap_data = functional.format_bitmap(bitmap, crop)
                logger.debug(f"smart_segmentation inference done!")
                response = {
                    "origin": bitmap_origin,
                    "bitmap": bitmap_data,
                    "success": True,
                    "error": None,
                }
            else:
                logger.debug(f"Predicted mask is empty.")
                response = {
                    "origin": None,
                    "bitmap": None,
                    "success": True,
                    "error": None,
                }
            return response
