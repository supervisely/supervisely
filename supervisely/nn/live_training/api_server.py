"""FastAPI surface for the live-training app.

Every endpoint that used to live in the standalone orchestrator at
``src/main.py`` is now served directly by the trainer through this module.
The endpoint handlers stay framework-agnostic — anything model-specific
(predictions, key-frame embeddings) is routed through the
``LiveTraining`` subclass via hooks.
"""

import asyncio
import concurrent.futures
import threading
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI, Request, Response

import supervisely as sly
from supervisely import logger
from supervisely.annotation.tag_meta import TagTargetType
from supervisely.api.module_api import ApiField

from .request_queue import RequestType
from .utils import TrainingStoppedException
from .video_utils import (
    create_video_object,
    filter_objects_by_confidence,
    get_uniform_frame_indices,
    label_to_video_figure_json,
    load_uniform_video_frames,
    refresh_meta,
    upload_video_figures,
)

if TYPE_CHECKING:
    from .live_training import LiveTraining


def start_api_server(
    lt: "LiveTraining", host: str = "0.0.0.0", port: int = 8000
) -> threading.Thread:
    """Start FastAPI server in a daemon thread, bound to ``lt``'s state."""
    server_app = lt.app.get_server()
    create_api(server_app, lt)

    config = uvicorn.Config(lt.app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True, name="APIServer")
    thread.start()

    logger.debug(f"Live Training API server started: http://{host}:{port}")
    return thread


def create_api(app: FastAPI, lt: "LiveTraining") -> FastAPI:

    @app.post("/start")
    async def start(response: Response):
        future = lt.request_queue.put(RequestType.START)
        return await _wait_for_result(future, response, timeout=None)

    @app.post("/predict")
    async def predict(request: Request, response: Response):
        sly_api = _api_from_request(request)
        state = request.state.state
        img_np = sly_api.image.download_np(state["image_id"])
        future = lt.request_queue.put(
            RequestType.PREDICT,
            {"image": img_np, "image_id": state["image_id"]},
        )
        return await _wait_for_result(future, response)

    @app.post("/predict-batch")
    async def predict_batch(request: Request, response: Response):
        sly_api = _api_from_request(request)
        state = request.state.state
        img_nps = sly_api.image.download_nps(state["image_ids"])
        future = lt.request_queue.put(
            RequestType.PREDICT_BATCH,
            {"images": img_nps, "image_ids": state["image_ids"]},
        )
        return await _wait_for_result(future, response)

    @app.post("/predict-video")
    async def predict_video(request: Request, response: Response):
        """Run inference on a single video frame and upload figures.

        Returns ``{"created_figure_ids": [...]}``.
        """
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state["video_id"]
        frame_index = state["frame_index"]
        confidence_threshold = state.get("confidence_threshold", 0.3)
        object_id = state.get("object_id")
        toolbox_session_id = state.get("toolbox_session_id")

        video_info = _resolve_video_info(lt, sly_api, video_id)
        frame_np = sly_api.video.frame.download_np(video_id, frame_index)
        frame_id = f"{video_id}_{frame_index}"

        future = lt.request_queue.put(
            RequestType.PREDICT,
            {"image": frame_np, "image_id": frame_id},
        )
        result = await _wait_for_result(future, response)
        if response.status_code >= 400:
            return result

        labels = filter_objects_by_confidence(
            result["objects"], lt.project_meta, confidence_threshold
        )
        if not labels:
            return {"created_figure_ids": []}

        if not object_id:
            object_id = create_video_object(sly_api, video_info, labels[0].obj_class)

        figures_json = [
            label_to_video_figure_json(label, object_id, frame_index) for label in labels
        ]
        figure_ids = upload_video_figures(sly_api, video_id, figures_json, toolbox_session_id)
        return {"created_figure_ids": figure_ids}

    @app.post("/predict-video-batch")
    async def predict_video_batch(request: Request, response: Response):
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state["video_id"]
        frame_indices = state["frame_indices"]
        frame_nps = sly_api.video.frame.download_nps(video_id, frame_indices)
        frame_ids = [f"{video_id}_{frame_index}" for frame_index in frame_indices]
        future = lt.request_queue.put(
            RequestType.PREDICT_BATCH,
            {"images": frame_nps, "image_ids": frame_ids},
        )
        return await _wait_for_result(future, response)

    @app.post("/add-sample")
    async def add_sample(request: Request, response: Response):
        sly_api = _api_from_request(request)
        state = request.state.state
        img_np = sly_api.image.download_np(state["image_id"])
        ann_json = sly_api.annotation.download_json(state["image_id"])
        img_info = sly_api.image.get_info_by_id(state["image_id"])
        future = lt.request_queue.put(
            RequestType.ADD_SAMPLE,
            {
                "image": img_np,
                "annotation": ann_json,
                "image_id": state["image_id"],
                "image_name": img_info.name,
            },
        )
        return await _wait_for_result(future, response)

    @app.post("/add-sample-video")
    async def add_sample_video(request: Request, response: Response):
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state["video_id"]
        frame_index = state["frame_index"]
        frame_np = sly_api.video.frame.download_np(video_id, frame_index)
        video_ann_json = sly_api.video.annotation.download(video_id)
        future = lt.request_queue.put(
            RequestType.ADD_SAMPLE_VIDEO,
            {
                "video_id": video_id,
                "frame_index": frame_index,
                "frame_np": frame_np,
                "video_ann_json": video_ann_json,
            },
        )
        return await _wait_for_result(future, response)

    @app.post("/add-samples-video")
    async def add_samples_video(request: Request, response: Response):
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state["video_id"]
        frame_indices = state["frame_indices"]
        frame_nps = sly_api.video.frame.download_nps(video_id, frame_indices)
        video_ann_json = sly_api.video.annotation.download(video_id)
        future = lt.request_queue.put(
            RequestType.ADD_SAMPLES_VIDEO,
            {
                "video_id": video_id,
                "frame_indices": frame_indices,
                "frame_nps": frame_nps,
                "video_ann_json": video_ann_json,
            },
        )
        return await _wait_for_result(future, response)

    @app.post("/highlight_key_frames")
    async def highlight_key_frames(request: Request, response: Response):
        """Tag uniform frames with ``need_to_label`` immediately, then prune
        in the background to the cluster medoids picked by the trainer."""
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state.get("video_id")
        toolbox_session_id = state.get("toolbox_session_id")

        video_info = _resolve_video_info(lt, sly_api, video_id)
        uniform_indices = get_uniform_frame_indices(video_id, sly_api)

        video_tag_meta = sly.TagMeta(
            name="need_to_label",
            value_type=sly.TagValueType.NONE,
            target_type=TagTargetType.FRAME_BASED,
        )
        new_tag_meta, lt.project_meta = refresh_meta(
            sly_api, video_info.project_id, lt.project_meta, video_tag_meta
        )

        tags_json = [
            {
                ApiField.TAG_ID: new_tag_meta.sly_id,
                ApiField.FRAME_RANGE: [idx, idx],
                ApiField.IS_FINISHED: True,
                ApiField.NON_FINAL_VALUE: False,
            }
            for idx in uniform_indices
        ]
        if toolbox_session_id is not None:
            sly_api.headers["x-toolbox-session-id"] = toolbox_session_id
        resp = sly_api.post(
            "videos.tags.bulk.add",
            {ApiField.VIDEO_ID: video_id, ApiField.TAGS: tags_json},
        )
        uniform_tag_ids = [obj[ApiField.ID] for obj in resp.json()]

        # Cancel any in-flight keyframe job and start a fresh one.
        if lt._keyframe_thread is not None and lt._keyframe_thread.is_alive():
            lt._keyframe_cancel.set()
            lt._keyframe_thread.join(timeout=1)
        lt._keyframe_cancel = threading.Event()
        cancel_event = lt._keyframe_cancel

        def _sample_and_prune():
            try:
                frames = load_uniform_video_frames(video_id, sly_api, uniform_indices)
                if cancel_event.is_set():
                    return
                future = lt.request_queue.put(RequestType.KEY_FRAMES, {"images": frames})
                key_result = future.result(timeout=600)
                if cancel_event.is_set():
                    return
                keep_set = set(key_result["indices"])
                for pos, tag_id in enumerate(uniform_tag_ids):
                    if cancel_event.is_set():
                        return
                    if pos not in keep_set:
                        sly_api.post("videos.tags.remove", {ApiField.ID: tag_id})
            except Exception as e:
                logger.warning(f"failed to sample key frames for video {video_id}: {e}")

        lt._keyframe_thread = threading.Thread(
            target=_sample_and_prune, daemon=True, name="KeyFrameSampler"
        )
        lt._keyframe_thread.start()
        return {"result": "successfully highlighted key frames"}

    # @app.post("/tracking_by_detection")
    # async def tracking_by_detection(request: Request, response: Response):
    #     if not lt.ready_to_predict:
    #         response.status_code = 409
    #         return _error_response_message("Model is not ready to produce any predictions yet")

    #     sly_api = _api_from_request(request)
    #     state = request.state.state
    #     video_id = state["video_id"]
    #     video_info = sly_api.video.get_info_by_id(video_id)

    #     start_frame = state.get("start_frame_index", 0)
    #     end_frame = state.get("end_frame_index", video_info.frames_count)
    #     step = state.get("step", 1)
    #     batch_size = state.get("batch_size", 8)
    #     confidence_threshold = state.get("confidence_threshold", 0.3)
    #     track_id = state.get("track_id")

    #     frame_indices = list(range(start_frame, end_frame, step))

    #     from supervisely.nn.tracker import BotSortTracker

    #     if lt._tracker is None:
    #         lt._tracker = BotSortTracker(device="cuda:0")
    #     lt._tracker.reset()

    #     if lt._tracker_thread is not None and lt._tracker_thread.is_alive():
    #         lt._tracker_cancel.set()
    #         lt._tracker_thread.join(timeout=1)
    #     lt._tracker_cancel = threading.Event()
    #     cancel_event = lt._tracker_cancel

    #     def _run():
    #         try:
    #             _run_tracking_by_detection(
    #                 lt=lt,
    #                 api=sly_api,
    #                 video_id=video_id,
    #                 video_info=video_info,
    #                 frame_indices=frame_indices,
    #                 batch_size=batch_size,
    #                 confidence_threshold=confidence_threshold,
    #                 track_id=track_id,
    #                 cancel_event=cancel_event,
    #             )
    #         except Exception as e:
    #             logger.warning(f"tracking_by_detection failed for video {video_id}: {e}")

    #     lt._tracker_thread = threading.Thread(target=_run, daemon=True, name="TrackingByDetection")
    #     lt._tracker_thread.start()
    #     return {"message": "Track task started."}

    @app.post("/inference_video_id")
    async def inference_video_id(request: Request, response: Response):
        if not lt.ready_to_predict:
            response.status_code = 409
            return _error_response_message("Model is not ready to produce any predictions yet")

        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state.get("videoId") or state.get("video_id")
        video_info = sly_api.video.get_info_by_id(video_id)

        start_frame = state.get("startFrameIndex", state.get("start_frame_index", 0))
        frames_count = state.get("framesCount", state.get("frames_count"))
        end_frame = state.get("endFrameIndex", state.get("end_frame_index"))
        direction_str = state.get("framesDirection", state.get("direction", "forward"))
        direction = 1 if direction_str == "forward" else -1
        step = state.get("stride", state.get("step", 1))
        batch_size = state.get("batch_size") or 5
        confidence_threshold = state.get("confidence_threshold", 0.3)

        if frames_count is not None:
            n_frames = frames_count
        elif end_frame is not None:
            n_frames = end_frame - start_frame
        else:
            n_frames = video_info.frames_count - start_frame

        frame_indices = list(
            range(start_frame, start_frame + direction * n_frames, direction * step)
        )

        predictions = await asyncio.to_thread(
            _run_inference_video,
            lt=lt,
            api=sly_api,
            video_info=video_info,
            frame_indices=frame_indices,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
        )
        return {"ann": predictions}

    @app.post("/get_session_info")
    async def get_session_info():
        return lt.get_session_info()

    @app.post("/is_deployed")
    async def is_deployed():
        return lt.deployment_info()

    @app.post("/get_custom_inference_settings")
    async def get_custom_inference_settings():
        return lt.get_custom_inference_settings()

    @app.post("/get_output_classes_and_tags")
    async def get_output_classes_and_tags():
        return lt.get_output_classes_and_tags()

    @app.post("/status")
    async def status():
        # /status bypasses the queue so it never blocks behind long-running
        # ADD_SAMPLES_VIDEO or PREDICT_BATCH work.
        return lt.status()

    return app


def _resolve_video_info(lt: "LiveTraining", api: sly.Api, video_id: int):
    if lt.video_info is None or lt.video_info.id != video_id:
        lt.video_info = api.video.get_info_by_id(video_id)
    return lt.video_info


# def _run_tracking_by_detection(
#     lt,
#     api,
#     video_id,
#     video_info,
#     frame_indices,
#     batch_size,
#     confidence_threshold,
#     track_id,
#     cancel_event,
# ):
#     img_size = (video_info.frame_height, video_info.frame_width)
#     track_id_to_obj_id = {}

#     for batch_indices in sly.batched(frame_indices, batch_size=batch_size):
#         if cancel_event.is_set():
#             return
#         batch_indices = list(batch_indices)
#         frame_nps = api.video.frame.download_nps(video_id, batch_indices)
#         frame_ids = [f"{video_id}_{idx}" for idx in batch_indices]

#         future = lt.request_queue.put(
#             RequestType.PREDICT_BATCH,
#             {"images": frame_nps, "image_ids": frame_ids},
#         )
#         batch_result = future.result(timeout=600)
#         if cancel_event.is_set():
#             return

#         figures_json = []
#         for frame_idx, frame, objects_json in zip(
#             batch_indices, frame_nps, batch_result["objects_batch"]
#         ):
#             labels = filter_objects_by_confidence(
#                 objects_json, lt.project_meta, confidence_threshold
#             )
#             ann = sly.Annotation(img_size=img_size, labels=labels)

#             matches = lt._tracker.update(frame, ann)
#             for match in matches:
#                 tracker_track_id = match["track_id"]
#                 label = match["label"]
#                 obj_id = track_id_to_obj_id.get(tracker_track_id)
#                 if obj_id is None:
#                     obj_id = create_video_object(api, video_info, label.obj_class)
#                     track_id_to_obj_id[tracker_track_id] = obj_id

#                 figure_json = label_to_video_figure_json(label, obj_id, frame_idx)
#                 if track_id is not None:
#                     figure_json[ApiField.TRACK_ID] = track_id
#                 figures_json.append(figure_json)

#         if figures_json:
#             upload_video_figures(api, video_id, figures_json)


def _run_inference_video(
    lt,
    api,
    video_info,
    frame_indices,
    batch_size,
    confidence_threshold,
):
    img_size = (video_info.frame_height, video_info.frame_width)
    predictions = []

    for batch_indices in sly.batched(frame_indices, batch_size=batch_size):
        batch_indices = list(batch_indices)
        frame_nps = api.video.frame.download_nps(video_info.id, batch_indices)
        frame_ids = [f"{video_info.id}_{idx}" for idx in batch_indices]
        future = lt.request_queue.put(
            RequestType.PREDICT_BATCH,
            {"images": frame_nps, "image_ids": frame_ids},
        )
        batch_result = future.result(timeout=600)
        for objects_json in batch_result["objects_batch"]:
            labels = filter_objects_by_confidence(
                objects_json, lt.project_meta, confidence_threshold
            )
            ann = sly.Annotation(img_size=img_size, labels=labels)
            predictions.append({"annotation": ann.to_json()})

    return predictions


async def _wait_for_result(
    future: concurrent.futures.Future,
    response: Response,
    timeout: float = 600.0,
):
    """Await a queue future on the event loop."""
    try:
        result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
    except asyncio.TimeoutError:
        response.status_code = 500
        result = _error_response_message("Request timeout - training may be busy")
    except TrainingStoppedException as e:
        response.status_code = 404
        result = _error_response_message(str(e))
    except Exception as e:
        response.status_code = 500
        result = _error_response_message(str(e))
    return result


def _api_from_request(request: Request) -> sly.Api:
    api = None
    try:
        api = request.state.api
    finally:
        if not isinstance(api, sly.Api):
            logger.warning(
                "sly.Api instance not found in request.state.api. "
                "Creating API from app's credentials."
            )
            api = sly.Api()
    return api


def _error_response_message(message: str):
    return {
        "details": {
            "message": message,
            "checkLogs": False,
        }
    }
