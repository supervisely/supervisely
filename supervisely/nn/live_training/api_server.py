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
from typing import TYPE_CHECKING, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response

import supervisely as sly
from supervisely import logger
from supervisely.annotation.tag_meta import TagTargetType
from supervisely.api.module_api import ApiField

from .request_queue import RequestType
from .utils import TrainingStoppedException
from .video_utils import (
    create_video_objects,
    filter_objects_by_confidence,
    get_uniform_frame_indices,
    label_to_video_figure_json,
    refresh_meta,
    remove_video_figures,
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
        """Predict on the start frame; if the InputNumber widget is > 1,
        also propagate detections forward with BotSort in a background
        thread.

        Returns ``{"created_figure_ids": [...]}`` for the start frame
        immediately. Subsequent frames are uploaded asynchronously.
        """
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state["video_id"]
        frame_index = state["frame_index"]
        confidence_threshold = state.get("confidence_threshold", 0.3)
        object_id = state.get("object_id")
        toolbox_session_id = state.get("toolbox_session_id")
        track_id = state.get("track_id")
        n_frames = state.get("n_frames")
        if n_frames:
            n_frames = max(1, int(n_frames))
            logger.info(
                f"[predict-video] video_id={video_id} frame_index={frame_index} "
                f"n_frames={n_frames} (from request state)"
            )
        else:
            raw_widget_value = lt.tracking_frames_widget.get_value()
            n_frames = max(1, int(raw_widget_value))
            logger.debug(
                f"[predict-video] video_id={video_id} frame_index={frame_index} "
                f"n_frames={n_frames} (from widget value {raw_widget_value!r})"
            )

        video_info = _resolve_video_info(lt, sly_api, video_id)
        if lt.frame_cache is not None:
            lt.frame_cache.ensure_window(video_id, frame_index, sly_api)
        frame_0_np = _get_video_frame(lt, sly_api, video_id, frame_index)
        frame_id = f"{video_id}_{frame_index}"

        future = lt.request_queue.put(
            RequestType.PREDICT,
            {"image": frame_0_np, "image_id": frame_id},
        )
        result = await _wait_for_result(future, response)
        if response.status_code is not None and response.status_code >= 400:
            return result

        labels_0 = filter_objects_by_confidence(
            result["objects"], lt.project_meta, confidence_threshold
        )
        logger.info(
            f"[predict-video] frame {frame_index}: model returned "
            f"{len(result.get('objects', []))} raw, "
            f"{len(labels_0)} after conf>={confidence_threshold}"
        )
        if not labels_0:
            return {"created_figure_ids": []}

        # Downloaded once (lazily) and shared between object-id resolution and
        # the background tracker; ``annotation.download`` always pulls the whole
        # video annotation, so we avoid doing it 2-3x per request.
        video_ann_json = None
        # ``labels_0`` / ``object_ids_0`` seed the forward tracker (when N>1);
        # ``upload_labels`` / ``upload_object_ids`` are what we actually upload
        # to the start frame. They differ only when some predictions already
        # have a figure on the start frame: those must not be re-uploaded there,
        # but must still seed the tracker so the instance is carried forward.
        if n_frames == 1 and object_id is not None:
            object_ids_0 = [object_id] * len(labels_0)
            upload_labels = labels_0
            upload_object_ids = object_ids_0
        else:
            video_ann_json = sly_api.video.annotation.download(video_id)
            # Inherit object_ids from frame K-1's labels (same class, IoU > 0.5)
            # so re-predicting on consecutive frames doesn't keep creating new
            # VideoObjects for what's effectively the same tracked instance.
            # Predictions that already have a matching figure on frame K are
            # flagged as duplicates (is_dup) — kept for tracking, excluded from
            # the start-frame upload.
            resolved = _resolve_object_ids_for_predictions(
                sly_api,
                video_id=video_id,
                frame_index=frame_index,
                labels=labels_0,
                video_info=video_info,
                project_meta=lt.project_meta,
                video_ann_json=video_ann_json,
            )
            seed = [
                (label, oid)
                for label, (oid, _) in zip(labels_0, resolved)
                if oid is not None
            ]
            upload = [
                (label, oid)
                for label, (oid, is_dup) in zip(labels_0, resolved)
                if oid is not None and not is_dup
            ]
            if not seed:
                logger.info(
                    f"[predict-video] frame {frame_index}: no usable "
                    f"predictions, nothing to track or upload"
                )
                return {"created_figure_ids": []}
            labels_0 = [label for label, _ in seed]
            object_ids_0 = [oid for _, oid in seed]
            upload_labels = [label for label, _ in upload]
            upload_object_ids = [oid for _, oid in upload]
            if not upload_labels:
                logger.info(
                    f"[predict-video] frame {frame_index}: all predictions "
                    f"duplicate existing figures on the start frame; nothing "
                    f"to upload there, but tracking {len(seed)} forward"
                )

        if upload_labels:
            figures_0_json = [
                _label_to_video_figure_json_with_track(label, obj_id, frame_index, track_id)
                for label, obj_id in zip(upload_labels, upload_object_ids)
            ]
            figure_ids_0 = upload_video_figures(
                sly_api, video_id, figures_0_json, toolbox_session_id
            )
        else:
            figure_ids_0 = []

        if n_frames > 1:
            logger.info(
                f"[predict-video] N>1: spawning PredictVideoTracking for "
                f"frames {frame_index+1}..{frame_index+n_frames-1}"
            )
            if lt._predict_video_thread is not None and lt._predict_video_thread.is_alive():
                logger.info("[predict-video] cancelling in-flight PredictVideoTracking")
                lt._predict_video_cancel.set()
                lt._predict_video_thread.join(timeout=1)
            lt._predict_video_cancel = threading.Event()
            cancel_event = lt._predict_video_cancel

            def _run():
                try:
                    _continue_predict_video_tracking(
                        lt=lt,
                        api=sly_api,
                        video_id=video_id,
                        video_info=video_info,
                        frame_0_index=frame_index,
                        frame_0_np=frame_0_np,
                        labels_0=labels_0,
                        object_ids_0=object_ids_0,
                        n_frames=n_frames,
                        confidence_threshold=confidence_threshold,
                        toolbox_session_id=toolbox_session_id,
                        track_id=track_id,
                        cancel_event=cancel_event,
                        video_ann_json=video_ann_json,
                    )
                except Exception as e:
                    logger.warning(
                        f"predict-video tracking (video {video_id}, "
                        f"start {frame_index}, n={n_frames}) failed: {e}"
                    )

            lt._predict_video_thread = threading.Thread(
                target=_run, daemon=True, name="PredictVideoTracking"
            )
            lt._predict_video_thread.start()

        return {"created_figure_ids": figure_ids_0}

    @app.post("/predict-video-batch")
    async def predict_video_batch(request: Request, response: Response):
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state["video_id"]
        frame_indices = state["frame_indices"]
        frame_nps = _get_video_frames(lt, sly_api, video_id, frame_indices)
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

        context = (request.state.context if hasattr(request.state, "context") else {}) or {}
        toolbox_session_id = state.get("toolbox_session_id") or context.get("toolbox_session_id")

        # First add-sample-video for this video kicks off background frame
        # prefetch; later requests keep the sliding window moving (no-op in
        # FULL mode). Both run on background threads.
        if lt.frame_cache is not None:
            lt.frame_cache.start_prefetch(video_id, sly_api, current_index=frame_index)
            lt.frame_cache.ensure_window(video_id, frame_index, sly_api)

        # Download the video annotation once and share it between auto-track
        # (latency-critical) and the training ingest. ``download`` always pulls
        # the entire annotation, so doing it twice is wasteful.
        video_ann_json = await asyncio.to_thread(sly_api.video.annotation.download, video_id)

        # Kick off auto-track of the next frame FIRST so the MCITrack request
        # starts immediately, in parallel with the training ingest below.
        # Previously this was spawned only after the ingest future resolved,
        # which is gated behind a training iteration — tracked figures didn't
        # land on N+1 until then.
        #
        # Only used while the model isn't ready to predict — once it is, the
        # labeling UI calls /predict-video for the next frame itself (which does
        # its own previous-frame IoU inheritance). Running both paths would
        # upload two figures to N+1 (one from each).
        logger.info(
            f"[add-sample-video] video={video_id} frame={frame_index} "
            f"ready_to_predict={lt.ready_to_predict} "
            f"mcitrack_task_id={lt.mcitrack_task_id} "
            f"keyframes_uploaded={lt._keyframes_uploaded.is_set()} "
            f"toolbox_session_id={toolbox_session_id!r} "
            f"state_keys={list(state.keys())} "
            f"context_keys={list(context.keys())}"
        )
        if (
            lt.mcitrack_task_id is not None
            and not lt.ready_to_predict
            and not lt._keyframes_uploaded.is_set()
        ):

            def _auto_track():
                try:
                    lt.auto_track_next_frame(
                        video_id,
                        frame_index,
                        sly_api,
                        toolbox_session_id,
                        video_ann_json=video_ann_json,
                    )
                except Exception as e:
                    logger.warning(
                        f"auto-track of frame {frame_index + 1} " f"(video {video_id}) failed: {e}"
                    )

            threading.Thread(target=_auto_track, daemon=True, name="AutoTrackNextFrame").start()

        # Ingest the labeled frame into the training dataset off the response
        # path: the frame only needs to land in the dataset eventually, and
        # blocking here would stall the (already-running) auto-track too.
        def _enqueue():
            try:
                # Training-dataset frame — pin it so it survives until stop.
                frame_np = _get_video_frame(lt, sly_api, video_id, frame_index, pin=True)
                lt.request_queue.put(
                    RequestType.ADD_SAMPLE_VIDEO,
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "frame_np": frame_np,
                        "video_ann_json": video_ann_json,
                    },
                )
            except Exception as e:
                logger.warning(f"failed to enqueue add-sample-video for video {video_id}: {e}")

        threading.Thread(target=_enqueue, daemon=True, name="AddSampleVideo").start()

        return lt.status()

    @app.post("/add-samples-video")
    async def add_samples_video(request: Request, response: Response):
        """Fire-and-forget batch sample ingest.

        Downloading the frames + video annotation and processing them through
        the queue is too slow to block the caller on. Instead we:
          1. Compute the predicted status (with an optimistic phase flip if
             this batch will push us past ``initial_samples``).
          2. Spawn a background thread that downloads frames, downloads the
             video annotation, and queues ``ADD_SAMPLES_VIDEO``. Its future is
             intentionally discarded.
          3. Return the predicted status to the caller immediately.

        Mirrors ``src/main.py::add_samples`` (orchestrator) behaviour.
        """
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state["video_id"]
        frame_indices = state["frame_indices"]

        status = lt.status()
        # Phase.WAITING_FOR_SAMPLES is the string literal "waiting_for_samples";
        # importing Phase here would close an import cycle with live_training.py.
        if status["phase"] == "waiting_for_samples" and status["waiting_samples"] <= len(
            frame_indices
        ):
            status["phase"] = "initial_training"

        def _enqueue():
            try:
                # Training-dataset frames — pin them so they survive until stop.
                frame_nps = _get_video_frames(lt, sly_api, video_id, frame_indices, pin=True)
                video_ann_json = sly_api.video.annotation.download(video_id)
                lt.request_queue.put(
                    RequestType.ADD_SAMPLES_VIDEO,
                    {
                        "video_id": video_id,
                        "frame_indices": frame_indices,
                        "frame_nps": frame_nps,
                        "video_ann_json": video_ann_json,
                    },
                )
            except Exception as e:
                logger.warning(f"failed to enqueue add-samples-video for video {video_id}: {e}")

        threading.Thread(target=_enqueue, daemon=True, name="AddSamplesVideo").start()
        return status

    @app.post("/highlight_key_frames")
    async def highlight_key_frames(request: Request, response: Response):
        """Tag uniform frames with ``need_to_label`` immediately, then prune
        in the background to the cluster medoids picked by the trainer."""
        # Refuse before START arrives: any queued KEY_FRAMES request would be
        # rejected by ``_wait_for_start``, and we don't want to leave
        # uniform-index tags on the video for a job that can't complete.
        # Phase.READY_TO_START is the literal "ready_to_start" — importing
        # Phase here would close an import cycle with live_training.py.
        if lt.phase == "ready_to_start":
            response.status_code = 409
            return _error_response_message(
                "Live training is not started yet — send START before " "/highlight_key_frames."
            )

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

        # Tags are on the server now → the labeling UI's "finish and next"
        # will jump across key frames, so MCITrack tracking onto the
        # literal next frame is no longer useful. Block future
        # /add-sample-video auto-track spawns.
        lt._keyframes_uploaded.set()
        logger.info(
            f"[highlight_key_frames] uploaded {len(uniform_tag_ids)} uniform "
            f"need_to_label tags; auto-track on /add-sample-video disabled"
        )

        # Cancel any in-flight keyframe job and start a fresh one.
        if lt._keyframe_thread is not None and lt._keyframe_thread.is_alive():
            lt._keyframe_cancel.set()
            lt._keyframe_thread.join(timeout=1)
        lt._keyframe_cancel = threading.Event()
        cancel_event = lt._keyframe_cancel

        def _sample_and_prune():
            try:
                frames = _get_video_frames(lt, sly_api, video_id, uniform_indices)
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
        #
        # Exception: in "continue" mode, status requests received after the
        # START signal must wait until the checkpoint is fully restored,
        # otherwise the UI sees iter=0 / empty dataset while the trainer is
        # mid-restore. Requests received before START still pass through.
        if (
            lt.checkpoint_mode == "continue"
            and lt._start_received.is_set()
            and not lt._continue_checkpoint_loaded.is_set()
        ):
            await asyncio.to_thread(lt._continue_checkpoint_loaded.wait)
        # Hold the readiness signal until MCITrack can serve /track-api requests
        # (video + live-training only). On a cold machine MCITrack may take a
        # while to pull its docker image; reporting "ready to start" before then
        # means the first labeled frames wouldn't be auto-tracked. The event is
        # pre-set when MCITrack isn't expected, and set on boot failure too, so
        # this never blocks indefinitely.
        if not lt._mcitrack_boot_done.is_set():
            await asyncio.to_thread(lt._mcitrack_boot_done.wait)
        return lt.status()

    return app


def _resolve_video_info(lt: "LiveTraining", api: sly.Api, video_id: int):
    if lt.video_info is None or lt.video_info.id != video_id:
        lt.video_info = api.video.get_info_by_id(video_id)
    return lt.video_info


def _get_video_frame(
    lt: "LiveTraining", api: sly.Api, video_id: int, frame_index: int, pin: bool = False
) -> np.ndarray:
    """Fetch one video frame through the frame cache when available, falling
    back to a direct download."""
    if lt.frame_cache is not None:
        return lt.frame_cache.get_frame(video_id, frame_index, api, pin=pin)
    return api.video.frame.download_np(video_id, frame_index)


def _get_video_frames(
    lt: "LiveTraining", api: sly.Api, video_id: int, frame_indices: list, pin: bool = False
) -> list:
    """Fetch a batch of video frames through the frame cache when available,
    falling back to a direct batch download."""
    if lt.frame_cache is not None:
        return lt.frame_cache.get_frames(video_id, frame_indices, api, pin=pin)
    return api.video.frame.download_nps(video_id, frame_indices)


def _label_to_video_figure_json_with_track(
    label: sly.Label,
    object_id: int,
    frame_index: int,
    track_id: Optional[int],
) -> dict:
    fj = label_to_video_figure_json(label, object_id, frame_index)
    if track_id is not None:
        fj[ApiField.TRACK_ID] = track_id
    return fj


def _resolve_object_ids_for_predictions(
    api,
    video_id: int,
    frame_index: int,
    labels: list,
    video_info,
    project_meta: sly.ProjectMeta,
    video_ann_json: Optional[dict] = None,
) -> list:
    """For each label in ``labels``, return a ``(object_id, is_duplicate)`` tuple:

    * ``is_duplicate=True`` — a same-class Rectangle figure with IoU > 0.5
      already exists on ``frame_index``; ``object_id`` is that existing
      figure's object id. The caller must NOT re-upload a figure for it on
      ``frame_index`` (it would break ``_filter_annotation_video`` later), but
      may still use the pair to seed a forward tracker so the already-labeled
      instance is carried into the following frames.
    * ``is_duplicate=False`` — ``object_id`` to use for a fresh upload on
      ``frame_index``. Inherited from frame ``frame_index - 1``'s same-class
      Rectangle figures (IoU > 0.5) so object ids stay consistent across
      consecutive frames; or freshly created via ``create_video_objects`` if
      no inheritance match.

    Falls back to ``create_video_objects`` for every label when there's no
    previous frame.
    """
    from supervisely.metric.matching import get_geometries_iou

    if frame_index <= 0:
        logger.info(
            f"[predict-video] frame {frame_index}: no previous frame, "
            f"creating fresh VideoObjects for all {len(labels)} predictions"
        )
        fresh_ids = create_video_objects(api, video_info, [label.obj_class for label in labels])
        return [(oid, False) for oid in fresh_ids]

    if video_ann_json is None:
        video_ann_json = api.video.annotation.download(video_id)

    # obj_id -> class_name (built from the video's full objects list so
    # objects whose only figure is on prev frame are still resolvable).
    class_by_obj_id = {
        obj["id"]: project_meta.get_obj_class_by_id(obj["classId"]).name
        for obj in video_ann_json.get("objects", [])
        if "id" in obj and project_meta.get_obj_class_by_id(obj["classId"]) is not None
    }

    def _rect_figures_on(idx: int) -> list:
        """Return [(Rectangle, object_id, class_name), ...] for one frame."""
        fr = next(
            (f for f in video_ann_json.get("frames", []) if f.get("index") == idx),
            None,
        )
        out = []
        if fr is None:
            return out
        for fig_json in fr.get("figures", []):
            if fig_json.get("geometryType") != sly.Rectangle.geometry_name():
                continue
            obj_id = fig_json.get("objectId")
            class_name = class_by_obj_id.get(obj_id)
            if obj_id is None or class_name is None:
                continue
            out.append(
                (
                    sly.Rectangle.from_json(fig_json["geometry"]),
                    obj_id,
                    class_name,
                )
            )
        return out

    existing_on_k = _rect_figures_on(frame_index)
    prev_sources = _rect_figures_on(frame_index - 1)
    logger.info(
        f"[predict-video] frame {frame_index}: existing-on-K "
        f"{len(existing_on_k)} figures, prev frame {frame_index - 1} "
        f"{len(prev_sources)} figures"
    )

    # Pass 1: classify each label as drop (None), inherit (existing obj_id), or
    # new (needs a fresh VideoObject). New objects are created in a single bulk
    # call afterwards instead of one network round-trip per prediction.
    out_object_ids = [None] * len(labels)
    new_slots = []  # output indices that need a freshly-created object
    new_classes = []  # parallel obj_classes for the bulk create
    used_prev = set()
    for i, label in enumerate(labels):
        if not isinstance(label.geometry, sly.Rectangle):
            new_slots.append(i)
            new_classes.append(label.obj_class)
            continue
        pred_class = label.obj_class.name

        # 1. Duplicate-on-K filter: drop predictions that overlap an
        # existing same-class figure on the target frame.
        skip = False
        for ex_rect, ex_obj_id, ex_class in existing_on_k:
            if ex_class != pred_class:
                continue
            iou = get_geometries_iou(label.geometry, ex_rect)
            if iou > 0.5:
                logger.info(
                    f"[predict-video] pred {pred_class} dropped: "
                    f"duplicates existing object_id={ex_obj_id} on frame "
                    f"{frame_index} at IoU={iou:.3f}"
                )
                skip = True
                skip_obj_id = ex_obj_id
                break
        if skip:
            # Keep the existing figure's object_id so the caller can seed the
            # forward tracker with this instance, but flag it so the start
            # frame isn't given a duplicate figure.
            out_object_ids[i] = (skip_obj_id, True)
            continue

        # 2. Inherit object_id from frame K-1 by same-class IoU > 0.5.
        best_idx, best_obj_id, best_iou = -1, None, 0.5
        for idx, (rect, obj_id, class_name) in enumerate(prev_sources):
            if idx in used_prev:
                continue
            if class_name != pred_class:
                continue
            iou = get_geometries_iou(label.geometry, rect)
            if iou > best_iou:
                best_idx, best_obj_id, best_iou = idx, obj_id, iou
        if best_obj_id is None:
            new_slots.append(i)
            new_classes.append(label.obj_class)
            logger.info(
                f"[predict-video] pred {pred_class} unmatched "
                f"(best_iou={best_iou:.3f}); will create new object"
            )
        else:
            used_prev.add(best_idx)
            out_object_ids[i] = (best_obj_id, False)
            logger.info(
                f"[predict-video] pred {pred_class} matched prev "
                f"object_id={best_obj_id} at IoU={best_iou:.3f}"
            )

    # Pass 2: bulk-create all new objects in one request and scatter the ids
    # back into their slots.
    if new_slots:
        new_ids = create_video_objects(api, video_info, new_classes)
        for slot, new_id in zip(new_slots, new_ids):
            out_object_ids[slot] = (new_id, False)
    return out_object_ids


def _continue_predict_video_tracking(
    lt,
    api,
    video_id,
    video_info,
    frame_0_index: int,
    frame_0_np: np.ndarray,
    labels_0: list,
    object_ids_0: list,
    n_frames: int,
    confidence_threshold: float,
    toolbox_session_id: Optional[str],
    track_id: Optional[int],
    cancel_event: threading.Event,
    video_ann_json: Optional[dict] = None,
):
    from supervisely.nn.tracker import BotSortTracker

    logger.info(
        f"[predict-video tracker] entered: video={video_id} "
        f"frame_0={frame_0_index} n_frames={n_frames} "
        f"seed_labels={len(labels_0)} object_ids_0={object_ids_0}"
    )

    if lt._predict_video_tracker is None:
        logger.info("[predict-video tracker] constructing BotSortTracker(cuda:0)")
        lt._predict_video_tracker = BotSortTracker(device="cuda:0")
    tracker = lt._predict_video_tracker
    tracker.reset()

    img_size = (video_info.frame_height, video_info.frame_width)

    # Seed: feed frame 0 into the tracker and pair returned tracks back to
    # the object_ids we just created. Order-preserving on a freshly-reset
    # tracker (no prior tracks to confuse it).
    matches_0 = tracker.update(frame_0_np, sly.Annotation(img_size=img_size, labels=labels_0))
    track_id_to_obj_id = {m["track_id"]: oid for m, oid in zip(matches_0, object_ids_0)}
    logger.info(
        f"[predict-video tracker] seed: {len(matches_0)} matches_0 -> "
        f"track_id_to_obj_id={track_id_to_obj_id}"
    )

    frame_indices = list(range(frame_0_index + 1, frame_0_index + n_frames))
    frame_indices = [i for i in frame_indices if i < video_info.frames_count]
    logger.info(f"[predict-video tracker] will process frames={frame_indices}")
    if not frame_indices:
        return

    from supervisely.metric.matching import get_geometries_iou

    # Snapshot existing same-class Rectangle figures on every target frame so
    # we can replace overlapping ones with the model's predictions (otherwise
    # we'd leave duplicate figures behind, like the user just hit on frame 10).
    frame_indices_set = set(frame_indices)
    if video_ann_json is None:
        video_ann_json = api.video.annotation.download(video_id)
    class_by_obj_id = {
        obj["id"]: lt.project_meta.get_obj_class_by_id(obj["classId"]).name
        for obj in video_ann_json.get("objects", [])
        if "id" in obj
        and obj.get("classId") is not None
        and lt.project_meta.get_obj_class_by_id(obj["classId"]) is not None
    }
    existing_by_frame: dict = {}
    for fr in video_ann_json.get("frames", []):
        idx = fr.get("index")
        if idx not in frame_indices_set:
            continue
        out = []
        for fig_json in fr.get("figures", []):
            if fig_json.get("geometryType") != sly.Rectangle.geometry_name():
                continue
            obj_id = fig_json.get("objectId")
            fig_id = fig_json.get("id")
            class_name = class_by_obj_id.get(obj_id)
            if obj_id is None or fig_id is None or class_name is None:
                continue
            out.append(
                {
                    "rect": sly.Rectangle.from_json(fig_json["geometry"]),
                    "obj_id": obj_id,
                    "class": class_name,
                    "fig_id": fig_id,
                }
            )
        if out:
            existing_by_frame[idx] = out
    logger.info(
        f"[predict-video tracker] existing same-class Rectangle figures per "
        f"frame: { {k: len(v) for k, v in existing_by_frame.items()} }"
    )

    batch_size = 5
    for batch_indices in sly.batched(frame_indices, batch_size=batch_size):
        if cancel_event.is_set():
            logger.info("[predict-video tracker] cancelled before batch")
            return
        batch_indices = list(batch_indices)
        frame_nps = _get_video_frames(lt, api, video_id, batch_indices)
        frame_ids = [f"{video_id}_{idx}" for idx in batch_indices]

        future = lt.request_queue.put(
            RequestType.PREDICT_BATCH,
            {"images": frame_nps, "image_ids": frame_ids},
        )
        batch_result = future.result(timeout=600)
        if cancel_event.is_set():
            logger.info("[predict-video tracker] cancelled after PREDICT_BATCH")
            return

        figures_json = []
        figure_ids_to_remove = []

        # Pass A: run the tracker on each frame (must stay ordered) and collect
        # track_ids that don't yet have a VideoObject, so they can be created in
        # a single bulk request instead of one round-trip per new track.
        per_frame = []  # (frame_idx, matches, existing_here)
        new_track_ids = []  # first-seen order
        new_classes = []
        pending_track_ids = set()
        for frame_idx, frame, objects_json in zip(
            batch_indices, frame_nps, batch_result["objects_batch"]
        ):
            labels = filter_objects_by_confidence(
                objects_json, lt.project_meta, confidence_threshold
            )
            ann = sly.Annotation(img_size=img_size, labels=labels)

            matches = tracker.update(frame, ann)
            logger.info(
                f"[predict-video tracker] frame {frame_idx}: "
                f"{len(labels)} preds -> {len(matches)} matches"
            )
            for match in matches:
                t_id = match["track_id"]
                if t_id not in track_id_to_obj_id and t_id not in pending_track_ids:
                    pending_track_ids.add(t_id)
                    new_track_ids.append(t_id)
                    new_classes.append(match["label"].obj_class)
            per_frame.append((frame_idx, matches, existing_by_frame.get(frame_idx, [])))

        if new_track_ids:
            new_ids = create_video_objects(api, video_info, new_classes)
            for t_id, new_id in zip(new_track_ids, new_ids):
                track_id_to_obj_id[t_id] = new_id

        # Pass B: build the figure diff now that every match has an object_id.
        for frame_idx, matches, existing_here in per_frame:
            used_existing = set()
            for match in matches:
                tracker_track_id = match["track_id"]
                label = match["label"]
                obj_id = track_id_to_obj_id[tracker_track_id]

                pred_class = label.obj_class.name
                best_ex_i, best_iou = -1, 0.5
                for ex_i, ex in enumerate(existing_here):
                    if ex_i in used_existing or ex["class"] != pred_class:
                        continue
                    iou = get_geometries_iou(label.geometry, ex["rect"])
                    if iou > best_iou:
                        best_ex_i, best_iou = ex_i, iou
                if best_ex_i >= 0:
                    used_existing.add(best_ex_i)
                    ex = existing_here[best_ex_i]
                    figure_ids_to_remove.append(ex["fig_id"])
                    logger.info(
                        f"[predict-video tracker] frame {frame_idx}: "
                        f"replacing existing fig_id={ex['fig_id']} "
                        f"object_id={ex['obj_id']} with prediction "
                        f"object_id={obj_id} at IoU={best_iou:.3f}"
                    )

                figure_json = label_to_video_figure_json(label, obj_id, frame_idx)
                if track_id is not None:
                    figure_json[ApiField.TRACK_ID] = track_id
                figures_json.append(figure_json)

        if figure_ids_to_remove:
            logger.info(
                f"[predict-video tracker] removing {len(figure_ids_to_remove)} "
                f"existing figures: {figure_ids_to_remove}"
            )
            remove_video_figures(api, video_id, figure_ids_to_remove, toolbox_session_id)
        if figures_json:
            logger.info(
                f"[predict-video tracker] uploading {len(figures_json)} "
                f"figures for batch {batch_indices}"
            )
            upload_video_figures(api, video_id, figures_json, toolbox_session_id)


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
        frame_nps = _get_video_frames(lt, api, video_info.id, batch_indices)
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
