import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Union

from cacheout import Cache
from cachetools import LRUCache
from fastapi import Form, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse

from supervisely import Label, ObjClass, ProjectMeta
from supervisely import env as sly_env
from supervisely._utils import rand_str
from supervisely.app.content import get_data_dir
from supervisely.geometry.bitmap import Bitmap
from supervisely.imaging import image as sly_image
from supervisely.io.fs import silent_remove
from supervisely.nn.inference import Inference
from supervisely.nn.inference.interactive_segmentation import functional
from supervisely.nn.prediction_dto import PredictionSegmentation
from supervisely.sly_logger import logger

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
        _smart_cache_ttl = sly_env.smart_cache_ttl()
        _fast_cache_ttl = max(1, _smart_cache_ttl // 2)
        Inference.__init__(self, model_dir, custom_inference_settings, sliding_window_mode, use_gui)
        self._class_names = ["mask_prediction"]
        color = [255, 0, 0]
        self._model_meta = ProjectMeta([ObjClass(self._class_names[0], Bitmap, color)])
        self._inference_image_lock = threading.Lock()
        self._inference_image_cache = Cache(ttl=_fast_cache_ttl)
        self._init_mask_cache = LRUCache(maxsize=100)  # cache of sly.Bitmaps

        if not self._use_gui:
            try:
                self.load_on_device(model_dir, "cuda")
            except RuntimeError:
                self.load_on_device(model_dir, "cpu")
                logger.warn("Failed to load model on CUDA device.")

        logger.debug(
            "Smart cache params",
            extra={"ttl": _smart_cache_ttl, "maxsize": sly_env.smart_cache_size()},
        )

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
        geometry = Bitmap(dto.mask, extra_validation=False)
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

    def serve(self):
        super().serve()
        server = self._app.get_server()
        self.cache.add_cache_endpoint(server)

        @server.post("/smart_segmentation")
        def smart_segmentation(response: Response, request: Request):
            logger.debug(
                f"smart_segmentation inference: context=",
                extra={**request.state.context, "api_token": "***"},
            )

            # Parse request
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

            # Pre-process clicks
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

            # Download the image if is not in Cache
            app_dir = get_data_dir()
            hash_str = functional.get_hash_from_context(smtool_state)
            if hash_str not in self._inference_image_cache:
                logger.debug(f"downloading image: {hash_str}")
                image_np = functional.download_image_from_context(
                    smtool_state,
                    api,
                    app_dir,
                    self.cache.download_image,
                    self.cache.download_frame,
                    self.cache.download_image_by_hash,
                )
                self._inference_image_cache.set(hash_str, image_np)
            else:
                logger.debug(f"image found in cache: {hash_str}")
                image_np = self._inference_image_cache.get(hash_str)

            # Crop the image
            image_np = functional.crop_image(crop, image_np)
            image_path = os.path.join(app_dir, f"{time.time()}_{rand_str(10)}.jpg")
            sly_image.write(image_path, image_np)

            # Prepare init_mask (only for images)
            figure_id = smtool_state.get("figure_id")
            image_id = smtool_state.get("image_id")
            if smtool_state.get("init_figure") is True and image_id is not None:
                # Download and save in Cache
                init_mask = functional.download_init_mask(api, figure_id, image_id)
                self._init_mask_cache[figure_id] = init_mask
            elif self._init_mask_cache.get(figure_id) is not None:
                # Load from Cache
                init_mask = self._init_mask_cache[figure_id]
            else:
                init_mask = None
            if init_mask is not None:
                img_info = api.image.get_info_by_id(image_id)
                h, w = img_info.height, img_info.width
                init_mask = functional.bitmap_to_mask(init_mask, h, w)
                init_mask = functional.crop_image(crop, init_mask)
                assert init_mask.shape[:2] == image_np.shape[:2]
            settings["init_mask"] = init_mask

            # Predict
            self._inference_image_lock.acquire()
            try:
                logger.debug(f"predict: {smtool_state['request_uid']}")
                clicks_to_predict = [self.Click(c["x"], c["y"], c["is_positive"]) for c in clicks]
                pred_mask = self.predict(image_path, clicks_to_predict, settings).mask
            finally:
                logger.debug(f"predict done: {smtool_state['request_uid']}")
                self._inference_image_lock.release()
                silent_remove(image_path)

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

        @server.post("/smart_segmentation_batch")
        def smart_segmentation_batch(response: Response, request: Request):
            result = []
            logger.debug(
                f"smart_segmentation inference: context=",
                extra={**request.state.context, "api_token": "***"},
            )

            # Parse request
            try:
                state = request.state.state
                settings = self._get_inference_settings(state)
                api = request.state.api
                smtool_states = request.state.context.get("states", [])
            except Exception as exc:
                logger.warn("Error parsing request:" + str(exc), exc_info=True)
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "400: Bad request.", "success": False}

            for smtool_state in smtool_states:
                crop = smtool_state["crop"]
                positive_clicks, negative_clicks = (
                    smtool_state["positive"],
                    smtool_state["negative"],
                )
                if len(positive_clicks) + len(negative_clicks) == 0:
                    logger.warn("No clicks received.")
                    result.append(
                        {
                            "origin": None,
                            "bitmap": None,
                            "success": True,
                            "error": None,
                        }
                    )
                    continue

                # Pre-process clicks
                clicks = [{**click, "is_positive": True} for click in positive_clicks]
                clicks += [{**click, "is_positive": False} for click in negative_clicks]
                clicks = functional.transform_clicks_to_crop(crop, clicks)
                is_in_bbox = functional.validate_click_bounds(crop, clicks)
                if not is_in_bbox:
                    logger.warn(f"Invalid value: click is out of bbox bounds.")
                    result.append(
                        {
                            "origin": None,
                            "bitmap": None,
                            "success": True,
                            "error": None,
                        }
                    )
                    continue

                # Download the image if is not in Cache
                app_dir = get_data_dir()
                hash_str = functional.get_hash_from_context(smtool_state)
                if hash_str not in self._inference_image_cache:
                    logger.debug(f"downloading image: {hash_str}")
                    image_np = functional.download_image_from_context(
                        smtool_state,
                        api,
                        app_dir,
                        self.cache.download_image,
                        self.cache.download_frame,
                        self.cache.download_image_by_hash,
                    )
                    self._inference_image_cache.set(hash_str, image_np)
                else:
                    logger.debug(f"image found in cache: {hash_str}")
                    image_np = self._inference_image_cache.get(hash_str)

                # Crop the image
                image_np = functional.crop_image(crop, image_np)
                image_path = os.path.join(app_dir, f"{time.time()}_{rand_str(10)}.jpg")
                sly_image.write(image_path, image_np)

                # Prepare init_mask (only for images)
                figure_id = smtool_state.get("figure_id")
                image_id = smtool_state.get("image_id")
                if smtool_state.get("init_figure") is True and image_id is not None:
                    # Download and save in Cache
                    init_mask = functional.download_init_mask(api, figure_id, image_id)
                    self._init_mask_cache[figure_id] = init_mask
                elif self._init_mask_cache.get(figure_id) is not None:
                    # Load from Cache
                    init_mask = self._init_mask_cache[figure_id]
                else:
                    init_mask = None
                if init_mask is not None:
                    img_info = api.image.get_info_by_id(image_id)
                    h, w = img_info.height, img_info.width
                    init_mask = functional.bitmap_to_mask(init_mask, h, w)
                    init_mask = functional.crop_image(crop, init_mask)
                    assert init_mask.shape[:2] == image_np.shape[:2]
                settings["init_mask"] = init_mask

                # Predict
                self._inference_image_lock.acquire()
                try:
                    logger.debug(f"predict: {smtool_state['request_uid']}")
                    clicks_to_predict = [
                        self.Click(c["x"], c["y"], c["is_positive"]) for c in clicks
                    ]
                    pred_mask = self.predict(image_path, clicks_to_predict, settings).mask
                finally:
                    logger.debug(f"predict done: {smtool_state['request_uid']}")
                    self._inference_image_lock.release()
                    silent_remove(image_path)

                if pred_mask.any():
                    bitmap = Bitmap(pred_mask)
                    bitmap_origin, bitmap_data = functional.format_bitmap(bitmap, crop)
                    logger.debug(f"smart_segmentation inference done!")
                    result.append(
                        {
                            "origin": bitmap_origin,
                            "bitmap": bitmap_data,
                            "success": True,
                            "error": None,
                        }
                    )
                else:
                    logger.debug(f"Predicted mask is empty.")
                    result.append(
                        {
                            "origin": None,
                            "bitmap": None,
                            "success": True,
                            "error": None,
                        }
                    )
            return result

        @server.post("/smart_segmentation_files")
        def smart_segmentation_files(
            request: Request, files: List[UploadFile], settings: str = Form("{}")
        ):
            result = []
            settings = json.loads(settings)
            logger.debug(
                f"smart_segmentation inference: context=",
                extra={settings},
            )
            smtool_states = settings.get("state", [])
            inf_settings = self._get_inference_settings(settings)
            for file, smtool_state in zip(files, smtool_states):
                # Parse request
                try:
                    crop = smtool_state["crop"]
                    positive_clicks, negative_clicks = (
                        smtool_state["positive"],
                        smtool_state["negative"],
                    )
                    if len(positive_clicks) + len(negative_clicks) == 0:
                        logger.warn("No clicks received.")
                        result.append(
                            {
                                "origin": None,
                                "bitmap": None,
                                "success": True,
                                "error": None,
                            }
                        )
                        continue
                except Exception as exc:
                    logger.warn("Error parsing request:" + str(exc), exc_info=True)
                    return JSONResponse(
                        {"message": "400: Bad request.", "success": False}, status_code=400
                    )

                # Pre-process clicks
                clicks = [{**click, "is_positive": True} for click in positive_clicks]
                clicks += [{**click, "is_positive": False} for click in negative_clicks]
                clicks = functional.transform_clicks_to_crop(crop, clicks)
                is_in_bbox = functional.validate_click_bounds(crop, clicks)
                if not is_in_bbox:
                    logger.warn(f"Invalid value: click is out of bbox bounds.")
                    result.append(
                        {
                            "origin": None,
                            "bitmap": None,
                            "success": True,
                            "error": None,
                        }
                    )
                    continue

                # Crop the image
                app_dir = get_data_dir()
                image_np = sly_image.read_bytes(file.file.read())
                hash_str = functional.get_hash_from_context(smtool_state)
                self._inference_image_cache.set(hash_str, image_np)
                image_np = functional.crop_image(crop, image_np)
                image_path = os.path.join(app_dir, f"{time.time()}_{rand_str(10)}.jpg")
                sly_image.write(image_path, image_np)

                # Prepare init_mask (only for images)
                geom_data = smtool_state.get("geometry")
                init_mask = Bitmap.from_json(geom_data) if geom_data is not None else None
                if init_mask is not None:
                    h, w = image_np.shape[:2]
                    init_mask = functional.bitmap_to_mask(init_mask, h, w)
                    init_mask = functional.crop_image(crop, init_mask)
                    assert init_mask.shape[:2] == image_np.shape[:2]
                inf_settings["init_mask"] = init_mask

                # Predict
                self._inference_image_lock.acquire()
                try:
                    logger.debug(f"predict: {smtool_state['request_uid']}")
                    clicks_to_predict = [
                        self.Click(c["x"], c["y"], c["is_positive"]) for c in clicks
                    ]
                    pred_mask = self.predict(image_path, clicks_to_predict, settings).mask
                finally:
                    logger.debug(f"predict done: {smtool_state['request_uid']}")
                    self._inference_image_lock.release()
                    silent_remove(image_path)

                if pred_mask.any():
                    bitmap = Bitmap(pred_mask)
                    bitmap_origin, bitmap_data = functional.format_bitmap(bitmap, crop)
                    logger.debug(f"smart_segmentation inference done!")
                    result.append(
                        {
                            "origin": bitmap_origin,
                            "bitmap": bitmap_data,
                            "success": True,
                            "error": None,
                        }
                    )
                else:
                    logger.debug(f"Predicted mask is empty.")
                    result.append(
                        {
                            "origin": None,
                            "bitmap": None,
                            "success": True,
                            "error": None,
                        }
                    )
            return result
