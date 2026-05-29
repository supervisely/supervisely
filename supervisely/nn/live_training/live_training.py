import os
import threading
from typing import List, Optional

import numpy as np
from .api_server import start_api_server
from .request_queue import RequestQueue, RequestType, Request
from .incremental_dataset import IncrementalDataset
from .helpers import ClassMap
from .evaluator import LiveEvaluator
import supervisely as sly
from supervisely import logger
from supervisely.app.widgets import Card, InputNumber
from supervisely.nn import TaskType
from datetime import datetime
import signal
import sys
import time
from .checkpoint_utils import resolve_checkpoint, save_state_json
from .artifacts_utils import upload_artifacts
from .loss_plateau_detector import LossPlateauDetector
from .utils import TrainingStoppedException, BackgroundRequestHandler, set_exception, set_result
from pathlib import Path


class Phase:
    """String constants describing the live-training lifecycle phases."""

    READY_TO_START = "ready_to_start"
    WAITING_FOR_SAMPLES = "waiting_for_samples"
    INITIAL_TRAINING = "initial_training"
    TRAINING = "training"


class LiveTraining:
    """Base implementation of an interactive/live training loop driven by requests (start/add sample/predict/status)."""

    from torch import nn  # pylint: disable=import-error

    task_type: str = None  # Should be set in subclass
    framework_name: str = None  # Should be set in subclass

    _task2geometries = {
        TaskType.OBJECT_DETECTION: [sly.Rectangle],
        TaskType.INSTANCE_SEGMENTATION: [sly.Bitmap, sly.Polygon],
        TaskType.SEMANTIC_SEGMENTATION: [sly.Bitmap, sly.Polygon],
    }

    def __init__(
        self,
        initial_samples: int = 2,
        filter_classes_by_task: bool = True,
    ):
        """
        :param initial_samples: Min samples before training.
        :type initial_samples: int
        :param filter_classes_by_task: Filter obj classes by task geometry.
        :type filter_classes_by_task: bool
        """
        from torch import nn  # pylint: disable=import-error

        self.initial_samples = initial_samples
        self.filter_classes_by_task = filter_classes_by_task
        if self.task_type is None and self.filter_classes_by_task:
            raise ValueError(
                "task_type must be set in subclass if filter_classes_by_task is set to True"
            )
        if self.framework_name is None:
            raise ValueError("framework_name must be set in subclass")

        self.project_id = sly.env.project_id()
        self.team_id = sly.env.team_id()
        self.task_id = sly.env.task_id(raise_not_found=False)
        self.tracking_frames_widget = InputNumber(value=1, min=1, max=300, step=1, controls=True)
        _layout_card = Card(
            title="Number of frames to track",
            content=self.tracking_frames_widget,
        )
        self.app = sly.Application(layout=_layout_card)

        @self.tracking_frames_widget.value_changed
        def _log_tracking_frames_changed(new_value):
            logger.info(
                f"[tracking_frames_widget] value_changed callback fired: "
                f"new_value={new_value!r}"
            )

        self.api = sly.Api()
        self.request_queue = RequestQueue()

        if os.getenv("DEVELOP_AND_DEBUG") and not sly.is_production():
            logger.info(
                f"🔧 Initializing Develop & Debug application for project {self.project_id}..."
            )
            sly.app.development.supervisely_vpn_network(action="up")
            debug_task = sly.app.development.create_debug_task(
                self.team_id, port="8000", project_id=self.project_id
            )
            self.task_id = debug_task["id"]
        self.phase = Phase.READY_TO_START
        self.iter = 0
        self._loss = None
        self._is_paused = False
        self._should_pause_after_continue = False
        self.initial_iters = 60  # TODO: remove later
        self.project_meta = self._fetch_project_meta(self.project_id)
        self.class_map = self._init_class_map(self.project_meta)
        self.dataset: IncrementalDataset = None
        self.model: nn.Module = None
        self.loss_plateau_detector = self._init_loss_plateau_detector()
        self.work_dir = "app_data"
        self.latest_checkpoint_path = f"{self.work_dir}/checkpoints/latest.pth"

        self.checkpoint_mode = os.getenv("modal.state.checkpointMode", "scratch")
        selected_task_id_env = os.getenv("modal.state.selectedExperimentTaskId")
        self.selected_experiment_task_id = (
            int(selected_task_id_env) if selected_task_id_env else None
        )

        self.training_start_time = None
        self._upload_in_progress = False

        self.evaluator = self.init_evaluator()
        self._upload_interval = 7200
        self._last_upload_time = None

        self._inactivity_timeout = 24 * 3600  # 24 hours in seconds
        self._last_activity_time = None
        self._background_request_handler: BackgroundRequestHandler = None

        # Video-aware endpoints (highlight_key_frames, /predict-video N>1)
        # run their long-running work on background threads owned by the
        # LiveTraining instance. They are protected by per-job cancel events.
        self.video_info = None
        self._predict_video_tracker = None
        self._predict_video_thread: Optional[threading.Thread] = None
        self._predict_video_cancel = threading.Event()
        self._keyframe_thread: Optional[threading.Thread] = None
        self._keyframe_cancel = threading.Event()

        # MCITrack auto-start: looked up or launched lazily in a background
        # thread so __init__ doesn't block. None until the thread succeeds;
        # auto_track_next_frame silently no-ops while it's None.
        self.workspace_id = sly.env.workspace_id()
        self.agent_id = sly.env.agent_id()
        self.mcitrack_module_id = 475
        self.mcitrack_task_id: Optional[int] = None
        self._mcitrack_ready = threading.Event()

        # Start the API server last so that every attribute touched by status()
        # (phase, iter, dataset, evaluator, ...) is already initialized before
        # the server can serve a /status call from another thread.
        self._api_thread = start_api_server(self)

        threading.Thread(target=self._start_mcitrack_app, daemon=True, name="MCITrackBoot").start()

        # from . import live_training_instance
        # live_training_instance = self  # for access from other modules

    @property
    def ready_to_predict(self):
        return self.iter > self.initial_iters

    def status(self):
        labeled_count = self.dataset.num_labeled_samples if self.dataset is not None else 0
        return {
            "phase": self.phase,
            "samples_count": len(self.dataset) if self.dataset is not None else 0,
            "waiting_samples": max(0, self.initial_samples - labeled_count),
            "task_type": self.task_type,
            "iteration": self.iter,
            "loss": self._loss,
            "training_paused": self._is_paused,
            "ready_to_predict": self.ready_to_predict,
            "initial_iters": self.initial_iters,
            "model_quality": self.evaluator.ema_value if self.evaluator else None,
        }

    def run(self):
        self.training_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._add_shutdown_callback()
        self._last_upload_time = time.time()
        self._last_activity_time = time.time()

        work_dir_path = Path(self.work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)
        model_meta_path = work_dir_path / "model_meta.json"
        sly.json.dump_json_file(self.project_meta.to_json(), str(model_meta_path))

        try:
            self.phase = Phase.READY_TO_START
            self._wait_for_start()
            if self.checkpoint_mode == "continue":
                self._run_continue()
            elif self.checkpoint_mode == "finetune":
                self._run_finetune()
            else:
                self._run_from_scratch()
        except Exception as e:
            if not sly.is_production():
                raise e
            else:
                logger.error(f"Live training failed: {e}", exc_info=True)
                self._process_requests_while_finishing(str(e))
                self._save_and_upload()

    def _run_from_scratch(self):
        self.phase = Phase.WAITING_FOR_SAMPLES
        self._wait_for_initial_samples()
        self._process_requests_while_initializing()
        self.train(checkpoint_path=None)

    def _run_continue(self):
        checkpoint_path, state = self._load_checkpoint()
        self.load_state(state)
        image_ids = state.get("image_ids", [])
        if image_ids:
            self._restore_dataset(image_ids)
        self._process_requests_while_initializing()
        self.train(checkpoint_path=checkpoint_path)

    def _run_finetune(self):
        checkpoint_path, _ = self._load_checkpoint()
        self.phase = Phase.WAITING_FOR_SAMPLES
        self._wait_for_initial_samples()
        self._process_requests_while_initializing()
        self.train(checkpoint_path=checkpoint_path)

    def _wait_for_start(self):
        request = self.request_queue.get()
        while request.type != RequestType.START:
            request.future.set_exception(
                Exception(f"Unexpected request {request.type} while waiting for START")
            )
            request = self.request_queue.get()
        # When START is received
        status = self.status()
        status["phase"] = Phase.WAITING_FOR_SAMPLES
        request.future.set_result(status)

    def _wait_until_samples_added(
        self,
        samples_needed: int,
        max_wait_time: int = None,
        count_fn=None,
    ):
        sleep_interval = 0.3
        elapsed_time = 0
        if count_fn is None:
            samples_before = len(self.dataset)

            def reached():
                return len(self.dataset) - samples_before >= samples_needed

        else:

            def reached():
                return count_fn() >= samples_needed

        while not reached():
            if max_wait_time is not None and elapsed_time >= max_wait_time:
                raise RuntimeError("Timeout waiting for samples")

            if not self.request_queue.is_empty():
                self._process_pending_requests()

            if self._should_upload_periodically():
                # TODO: Upload in background thread to avoid blocking requests from web UI
                logger.info(f"Periodic upload (interval: {self._upload_interval}s)")
                self._save_and_upload()
                self._last_upload_time = time.time()

            if self._should_stop():
                logger.warning(f"No activity for {self._inactivity_timeout / 3600:.1f} hours")
                self._save_and_upload()
                sys.exit(0)

            time.sleep(sleep_interval)
            elapsed_time += sleep_interval

    def _wait_for_initial_samples(self):
        if self.dataset.num_labeled_samples >= self.initial_samples:
            return

        self.phase = Phase.WAITING_FOR_SAMPLES
        self._is_paused = True

        samples_needed = self.initial_samples - self.dataset.num_labeled_samples
        logger.info(
            f"Waiting for {samples_needed} more labeled samples "
            f"({self.dataset.num_labeled_samples}/{self.initial_samples})"
        )
        self._wait_until_samples_added(
            samples_needed=self.initial_samples,
            max_wait_time=3600,
            count_fn=lambda: self.dataset.num_labeled_samples,
        )

        self._is_paused = False

    def _process_pending_requests(self):
        # Back to synchronous processing.
        self._stop_background_request_processing()

        requests = self.request_queue.get_all()
        if not requests:
            return

        for request in requests:
            try:
                if request.type == RequestType.PREDICT:
                    result = self._handle_predict(request.data)
                    request.future.set_result(result)
                    self._last_activity_time = time.time()

                elif request.type == RequestType.PREDICT_BATCH:
                    result = self._handle_predict_batch(request.data)
                    request.future.set_result(result)
                    self._last_activity_time = time.time()

                elif request.type == RequestType.ADD_SAMPLE:
                    result = self._handle_add_sample(request.data)
                    request.future.set_result(result)
                    self._last_activity_time = time.time()

                elif request.type == RequestType.ADD_SAMPLE_VIDEO:
                    result = self._handle_add_sample_video(request.data)
                    request.future.set_result(result)

                elif request.type == RequestType.ADD_SAMPLES_VIDEO:
                    result = self._handle_add_samples_video(request.data)
                    request.future.set_result(result)

                elif request.type == RequestType.KEY_FRAMES:
                    result = self._handle_key_frames(request.data)
                    request.future.set_result(result)
                    self._last_activity_time = time.time()

            except Exception as e:
                logger.error(f"Error processing request {request.type}: {e}", exc_info=True)
                request.future.set_exception(e)

    def train(self, checkpoint_path: str = None):
        """
        Main training loop. Implement framework-specific training logic here.
        Prepare model config, set hyperparameters and run training.
        Handle phases: initial training, training
        """
        raise NotImplementedError

    def predict(self, model: nn.Module, image_np, image_info) -> list:
        """
        Run inference on a single image and return predictions as a list of sly figures in json format.
        """
        raise NotImplementedError

    def predict_batch(self, model: nn.Module, image_nps, image_infos) -> list:
        """
        Run inference on a batch of images and return predictions as a list of lists of sly figures in json format.
        """
        raise NotImplementedError

    def compute_key_frame_indices(self, image_nps: List[np.ndarray]) -> List[int]:
        """
        Return indices of representative frames among ``image_nps`` for the
        /highlight_key_frames endpoint. Subclass picks the embedding & clustering
        strategy.
        """
        raise NotImplementedError

    def get_session_info(self) -> dict:
        """Metadata for the /get_session_info endpoint. Subclass may override
        to add framework-specific fields."""
        return {
            "app_name": getattr(self, "app_name", "Live Training"),
            "session_id": self.task_id,
            "videos_support": True,
            "async_video_inference_support": True,
            "tracking_on_videos_support": True,
            "tracking_algorithms": ["botsort"],
            "batch_inference_support": True,
            "max_batch_size": 5,
            "task_type": self.task_type,
        }

    def get_custom_inference_settings(self) -> dict:
        return {"settings": {"confidence_threshold": 0.3}}

    def get_output_classes_and_tags(self) -> dict:
        return self.project_meta.to_json()

    def deployment_info(self) -> dict:
        return {
            "deployed": bool(self.ready_to_predict),
            "description": "Model is ready to receive requests",
        }

    def _handle_predict(self, data: dict):
        image_np = data["image"]
        image_id = data["image_id"]
        image_info = {"id": image_id}
        model = self.model
        was_training = model.training
        model.eval()
        try:
            objects_raw = self.predict(self.model, image_np=image_np, image_info=image_info)

            if self.evaluator:
                image_shape = image_np.shape[:2]
                self.evaluator.store_prediction(image_id, objects_raw, image_shape)

            return {
                "objects": objects_raw,
                "image_id": image_id,
                "status": self.status(),
            }
        finally:
            # Restore training mode
            if was_training:
                model.train()

    def _handle_predict_batch(self, data: dict):
        image_nps = data["images"]
        image_ids = data["image_ids"]
        image_infos = [{"id": image_id} for image_id in image_ids]
        model = self.model
        was_training = model.training
        model.eval()
        try:
            objects_raw_batch = self.predict_batch(
                self.model, image_nps=image_nps, image_infos=image_infos
            )

            if self.evaluator:
                for image_id, image_np, objects_raw in zip(image_ids, image_nps, objects_raw_batch):
                    image_shape = image_np.shape[:2]
                    self.evaluator.store_prediction(image_id, objects_raw, image_shape)

            return {
                "objects_batch": objects_raw_batch,
                "image_ids": image_ids,
                "status": self.status(),
            }
        finally:
            # Restore training mode
            if was_training:
                model.train()

    def _handle_key_frames(self, data: dict) -> dict:
        image_nps = data["images"]
        model = self.model
        was_training = model.training if model is not None else False
        if model is not None:
            model.eval()
        try:
            indices = self.compute_key_frame_indices(image_nps)
            return {"indices": list(indices)}
        finally:
            if model is not None and was_training:
                model.train()

    def add_sample(
        self, image_id: int, image_np: np.ndarray, annotation: sly.Annotation, image_name: str
    ) -> dict:
        return self.dataset.add_or_update(image_id, image_np, annotation, image_name)

    def _handle_add_sample(self, data: dict):
        ann_json = data["annotation"]
        ann_json = self._filter_annotation(ann_json)
        sly_ann = sly.Annotation.from_json(ann_json, self.project_meta)
        image_id = data["image_id"]
        self.add_sample(
            image_id=image_id,
            image_np=data["image"],
            annotation=sly_ann,
            image_name=data["image_name"],
        )
        if not sly_ann.labels and self.phase == Phase.WAITING_FOR_SAMPLES:
            logger.debug(f"Added unlabeled sample {image_id}; not counted toward initial threshold")
        if self.evaluator and self.phase != Phase.WAITING_FOR_SAMPLES:
            result = self.evaluator.evaluate(image_id, sly_ann)
            if result is not None:
                metric_name = self.evaluator.metric_name
                logger.info(
                    f"Image {image_id}: {metric_name}={result['metric_value']:.3f}, "
                    f"EMA={result['ema_value']:.3f}"
                )

        if (
            self.dataset.num_labeled_samples >= self.initial_samples
        ) and self.phase == Phase.WAITING_FOR_SAMPLES:
            self.phase = Phase.INITIAL_TRAINING

        return {
            "image_id": image_id,
            "status": self.status(),
        }

    def add_sample_video(
        self,
        frame_id: str,
        frame_np: np.ndarray,
        annotation: sly.Annotation,
    ) -> dict:
        return self.dataset.add_or_update_video(frame_id, frame_np, annotation)

    def frame_ann_to_img_ann(
        self,
        frame_ann: sly.Frame,
        frame_h: int,
        frame_w: int,
    ):
        frame_figures = frame_ann.figures

        labels = []
        for figure in frame_figures:
            geometry = figure.geometry
            obj_class = figure.video_object.obj_class
            labels.append(sly.Label(geometry, obj_class))

        img_ann = sly.Annotation((frame_h, frame_w), labels)
        return img_ann

    def _handle_add_sample_video(self, data: dict):
        frame_index = data["frame_index"]
        frame_id = f"{data['video_id']}_{frame_index}"
        frame_np = data["frame_np"]
        video_ann_json = data["video_ann_json"]
        video_objects_json, frame_ann_json = self._filter_annotation_video(
            video_ann_json, frame_index
        )
        frame_h, frame_w = video_ann_json["size"]["height"], video_ann_json["size"]["width"]
        if frame_ann_json:
            key_id_map = sly.KeyIdMap()
            video_obj_col = sly.VideoObjectCollection.from_json(
                video_objects_json, self.project_meta, key_id_map
            )
            frames_count = video_ann_json["framesCount"]
            frame_ann = sly.Frame.from_json(frame_ann_json, video_obj_col, frames_count, key_id_map)
            img_ann = self.frame_ann_to_img_ann(frame_ann, frame_h, frame_w)
        else:
            img_ann = sly.Annotation((frame_h, frame_w), labels=[])
        self.add_sample_video(
            frame_id=frame_id,
            frame_np=frame_np,
            annotation=img_ann,
        )

        if not img_ann.labels and self.phase == Phase.WAITING_FOR_SAMPLES:
            logger.debug(f"Added unlabeled sample {frame_id}; not counted toward initial threshold")
        if self.evaluator and self.phase != Phase.WAITING_FOR_SAMPLES:
            # Seed predictions for this frame so EMA updates on every added
            # sample (otherwise the user must manually trigger predict on
            # the same frame for model_quality to move).
            if self.ready_to_predict:
                self._predict_and_store_for_evaluator(frame_id, frame_np)
            result = self.evaluator.evaluate(frame_id, img_ann)
            if result is not None:
                metric_name = self.evaluator.metric_name
                logger.info(
                    f"Image {frame_id}: {metric_name}={result['metric_value']:.3f}, "
                    f"EMA={result['ema_value']:.3f}"
                )

        if (
            self.dataset.num_labeled_samples >= self.initial_samples
        ) and self.phase == Phase.WAITING_FOR_SAMPLES:
            self.phase = Phase.INITIAL_TRAINING

        return {
            "image_id": frame_id,
            "status": self.status(),
        }

    def _handle_add_samples_video(self, data: dict):
        frame_indices = data["frame_indices"]
        frame_ids = [f"{data['video_id']}_{frame_index}" for frame_index in frame_indices]
        frame_nps = data["frame_nps"]
        video_ann_json = data["video_ann_json"]

        for frame_id, frame_index, frame_np in zip(frame_ids, frame_indices, frame_nps):
            video_objects_json, frame_ann_json = self._filter_annotation_video(
                video_ann_json, frame_index
            )
            frame_h, frame_w = video_ann_json["size"]["height"], video_ann_json["size"]["width"]
            if frame_ann_json:
                key_id_map = sly.KeyIdMap()
                video_obj_col = sly.VideoObjectCollection.from_json(
                    video_objects_json, self.project_meta, key_id_map
                )
                frames_count = video_ann_json["framesCount"]
                frame_ann = sly.Frame.from_json(
                    frame_ann_json, video_obj_col, frames_count, key_id_map
                )
                img_ann = self.frame_ann_to_img_ann(frame_ann, frame_h, frame_w)
            else:
                img_ann = sly.Annotation((frame_h, frame_w), labels=[])

            self.add_sample_video(
                frame_id=frame_id,
                frame_np=frame_np,
                annotation=img_ann,
            )

            if not img_ann.labels and self.phase == Phase.WAITING_FOR_SAMPLES:
                logger.debug(
                    f"Added unlabeled sample {frame_id}; not counted toward initial threshold"
                )
            if self.evaluator and self.phase != Phase.WAITING_FOR_SAMPLES:
                if self.ready_to_predict:
                    self._predict_and_store_for_evaluator(frame_id, frame_np)
                result = self.evaluator.evaluate(frame_id, img_ann)
                if result is not None:
                    metric_name = self.evaluator.metric_name
                    logger.info(
                        f"Image {frame_id}: {metric_name}={result['metric_value']:.3f}, "
                        f"EMA={result['ema_value']:.3f}"
                    )

            if (
                self.dataset.num_labeled_samples >= self.initial_samples
            ) and self.phase == Phase.WAITING_FOR_SAMPLES:
                self.phase = Phase.INITIAL_TRAINING

        return {
            "image_ids": frame_ids,
            "status": self.status(),
        }

    def _predict_and_store_for_evaluator(self, image_id, image_np):
        """Run a single-image prediction and stash it on the evaluator so
        the next ``evaluator.evaluate`` call has something to compare GT
        against. Toggles model.eval()/train() like ``_handle_predict``.
        """
        model = self.model
        if model is None:
            return
        was_training = model.training
        model.eval()
        try:
            objects_raw = self.predict(model, image_np=image_np, image_info={"id": image_id})
            self.evaluator.store_prediction(image_id, objects_raw, image_np.shape[:2])
        except Exception as e:
            logger.warning(f"failed to seed evaluator prediction for {image_id}: {e}")
        finally:
            if was_training:
                model.train()

    def _fetch_project_meta(self, project_id: int) -> sly.ProjectMeta:
        project_meta = self.api.project.get_meta(project_id)
        project_meta = sly.ProjectMeta.from_json(project_meta)
        return project_meta

    def _init_class_map(self, project_meta: sly.ProjectMeta) -> ClassMap:
        obj_classes = list(project_meta.obj_classes)

        if self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            obj_classes.insert(0, sly.ObjClass(name="_background_", geometry_type=sly.Bitmap))

        if self.filter_classes_by_task:
            allowed_geometries = self._task2geometries[self.task_type] + [sly.AnyGeometry]
            obj_classes = [
                obj_class
                for obj_class in obj_classes
                if obj_class.geometry_type in allowed_geometries
            ]

        return ClassMap(obj_classes)

    def _filter_annotation(self, ann_json: dict) -> dict:
        # Filter objects according to class_map
        # Important: Must be filtered before sly.Annotation.from_json due to static project meta
        allowed_geometries = self._task2geometries[self.task_type]
        allowed_geometries = [geom.geometry_name() for geom in allowed_geometries]
        filtered_objects = []
        for obj in ann_json["objects"]:
            sly_id = obj["classId"]
            if sly_id in self.class_map.sly_ids and obj["geometryType"] in allowed_geometries:
                filtered_objects.append(obj)
        ann_json["objects"] = filtered_objects
        return ann_json

    def _start_mcitrack_app(self):
        """Find or launch the MCITrack app and wait until it answers RPCs.

        Failure here is non-fatal: ``auto_track_next_frame`` silently no-ops
        while ``self.mcitrack_task_id is None``.
        """
        from supervisely.api.task_api import TaskApi

        try:
            sessions = self.api.app.get_sessions(
                self.team_id,
                self.mcitrack_module_id,
                statuses=[TaskApi.Status.STARTED],
            )
            if sessions:
                task_id = sessions[0].task_id
                logger.info(f"Reusing MCITrack session: task_id={task_id}")
            else:
                session_info = self.api.app.start(
                    agent_id=self.agent_id,
                    module_id=self.mcitrack_module_id,
                    workspace_id=self.workspace_id,
                )
                task_id = session_info.task_id
                logger.info(f"Started MCITrack session: task_id={task_id}")

            if not self.api.app.is_ready_for_api_calls(task_id):
                self.api.app.wait_until_ready_for_api_calls(task_id, attempts=100000)
            time.sleep(10)  # status flips before the HTTP server is actually up
            self.mcitrack_task_id = task_id
            self._mcitrack_ready.set()
            logger.info("MCITrack session is ready for /track-api requests")
        except Exception as e:
            logger.warning(
                f"Failed to start MCITrack (auto-track on add-sample-video "
                f"will be disabled): {e}"
            )

    def auto_track_next_frame(
        self,
        video_id: int,
        frame_index: int,
        api: sly.Api,
        toolbox_session_id: Optional[str] = None,
    ) -> None:
        """Propagate the labels on ``frame_index`` to ``frame_index + 1``
        with MCITrack and reconcile against existing figures there.

        Only runs while ``ready_to_predict`` is False — once the model can
        predict, the labeling UI's own /predict-video call on N+1 handles
        propagation (with previous-frame IoU inheritance) and we'd
        otherwise upload two figures to the same frame.

        Existing N+1 figures with same-class IoU > 0.5 against a tracked
        figure are removed and re-added with the source ``object_id`` so ids
        stay consistent across frames. Unmatched tracked figures are added
        at the tracked coordinates.
        """
        from .video_utils import remove_video_figures, upload_video_figures
        from supervisely.api.module_api import ApiField
        from supervisely.metric.matching import get_geometries_iou

        logger.info(
            f"[auto-track] entered: video={video_id} frame={frame_index} "
            f"mcitrack_task_id={self.mcitrack_task_id} "
            f"toolbox_session_id={toolbox_session_id!r}"
        )
        if self.mcitrack_task_id is None:
            logger.info("[auto-track] MCITrack not available, skipping")
            return

        # 1. Re-download the video annotation so we see the figure the user
        # just committed on frame N (and any concurrent edits on N+1).
        video_ann_json = api.video.annotation.download(video_id)
        next_frame_index = frame_index + 1
        if next_frame_index >= video_ann_json["framesCount"]:
            logger.info(
                f"[auto-track] next_frame {next_frame_index} >= "
                f"framesCount {video_ann_json['framesCount']}, skipping"
            )
            return

        # 2. obj_id -> class_name map from the video's top-level objects list.
        class_by_obj_id = {
            obj["id"]: self.project_meta.get_obj_class_by_id(obj["classId"]).name
            for obj in video_ann_json.get("objects", [])
            if "id" in obj and self.project_meta.get_obj_class_by_id(obj["classId"]) is not None
        }

        # 3. Source figures on frame N.
        src_frame_ann_json = next(
            (fr for fr in video_ann_json.get("frames", []) if fr.get("index") == frame_index),
            None,
        )
        if not src_frame_ann_json or not src_frame_ann_json.get("figures"):
            logger.info(f"[auto-track] no figures on frame {frame_index}, skipping")
            return

        # sources is parallel-indexed with input_geometries.
        sources = []  # list of (src_object_id, src_class_name)
        input_geometries = []
        for fig_json in src_frame_ann_json["figures"]:
            if fig_json.get("geometryType") != sly.Rectangle.geometry_name():
                continue
            src_object_id = fig_json.get("objectId")
            src_class = class_by_obj_id.get(src_object_id)
            if src_object_id is None or src_class is None:
                continue
            sources.append((src_object_id, src_class))
            input_geometries.append(
                {"type": sly.Rectangle.geometry_name(), "data": fig_json["geometry"]}
            )
        if not input_geometries:
            logger.info(f"[auto-track] no Rectangle sources on frame {frame_index}, skipping")
            return
        logger.info(f"[auto-track] frame {frame_index} sources: {sources}")

        # 4. Ask MCITrack for predictions on frame N+1.
        response = self.api.task.send_request(
            self.mcitrack_task_id,
            "track-api",
            {},
            context={
                "videoId": video_id,
                "frameIndex": frame_index,
                "frames": 1,
                "direction": "forward",
                "input_geometries": input_geometries,
            },
            timeout=120,
        )
        # Shape: [[geo_obj_0, geo_obj_1, ...]] — one inner list per frame.
        if not response or not response[0]:
            logger.info("[auto-track] MCITrack returned empty, skipping reconcile")
            return
        tracked_per_obj = response[0]
        tracked_sources = sources  # parallel by construction
        logger.info(
            f"[auto-track] tracked_per_obj count={len(tracked_per_obj)}, "
            f"tracked_sources={tracked_sources}"
        )

        # 5. Existing figures on frame N+1.
        dst_frame_ann_json = next(
            (fr for fr in video_ann_json.get("frames", []) if fr.get("index") == next_frame_index),
            None,
        )
        existing = []  # list of (figure_id, geometry_json, class_name)
        if dst_frame_ann_json:
            for fig_json in dst_frame_ann_json.get("figures", []):
                if fig_json.get("geometryType") != sly.Rectangle.geometry_name():
                    continue
                fig_id = fig_json.get("id")
                if fig_id is None:
                    # Figure not yet persisted server-side — skip the merge
                    # path; we'd have no id to delete.
                    continue
                obj_class_name = class_by_obj_id.get(fig_json.get("objectId"))
                existing.append((fig_id, fig_json["geometry"], obj_class_name))
        logger.info(
            f"[auto-track] frame {next_frame_index} existing figures: "
            f"count={len(existing)} ids={[e[0] for e in existing]} "
            f"classes={[e[2] for e in existing]}"
        )

        # 6. Greedy IoU match (same class only, strict > 0.5).
        tracked_rects = [sly.Rectangle.from_json(item["data"]) for item in tracked_per_obj]
        existing_rects = [sly.Rectangle.from_json(geo) for _, geo, _ in existing]

        figure_ids_to_remove = []
        figures_to_add = []  # list of (geometry_json, object_id)
        used_existing = set()
        for t_idx, t_rect in enumerate(tracked_rects):
            src_object_id, src_class = tracked_sources[t_idx]
            best_e_idx, best_iou = -1, 0.5
            for e_idx, e_rect in enumerate(existing_rects):
                if e_idx in used_existing:
                    continue
                if existing[e_idx][2] != src_class:
                    continue
                iou = get_geometries_iou(t_rect, e_rect)
                if iou > best_iou:
                    best_e_idx, best_iou = e_idx, iou
            if best_e_idx >= 0:
                used_existing.add(best_e_idx)
                fig_id, e_geo, _ = existing[best_e_idx]
                figure_ids_to_remove.append(fig_id)
                figures_to_add.append((e_geo, src_object_id))
            else:
                figures_to_add.append((tracked_per_obj[t_idx]["data"], src_object_id))

        logger.info(
            f"[auto-track] reconcile: remove {len(figure_ids_to_remove)} "
            f"existing figures (ids={figure_ids_to_remove}), "
            f"add {len(figures_to_add)} new with object_ids="
            f"{[oid for _, oid in figures_to_add]}"
        )

        # 7. Push the diff — remove first so re-added geometry doesn't collide.
        if figure_ids_to_remove:
            remove_video_figures(api, video_id, figure_ids_to_remove, toolbox_session_id)

        figures_json = [
            {
                ApiField.OBJECT_ID: object_id,
                ApiField.GEOMETRY_TYPE: sly.Rectangle.geometry_name(),
                ApiField.GEOMETRY: geometry_json,
                ApiField.META: {ApiField.FRAME: next_frame_index},
                ApiField.NN_CREATED: True,
                ApiField.NN_UPDATED: True,
            }
            for geometry_json, object_id in figures_to_add
        ]
        if figures_json:
            upload_video_figures(api, video_id, figures_json, toolbox_session_id)

    def _filter_annotation_video(self, video_ann_json: dict, frame_index: int) -> dict:
        # Filter objects according to class_map
        # Important: Must be filtered before sly.Annotation.from_json due to static project meta
        allowed_geometries = self._task2geometries[self.task_type]
        allowed_geometries = [geom.geometry_name() for geom in allowed_geometries]

        frame_ann_json = [
            frame for frame in video_ann_json["frames"] if frame["index"] == frame_index
        ]

        if not frame_ann_json:
            # raise ValueError(
            #     f"Input frame must be annotated, but frame with index {frame_index} does not contain labels"
            # )
            return None, None
        frame_ann_json = frame_ann_json[0]

        filtered_objects, filtered_figures = [], []
        seen_obj_ids = set()
        for figure in frame_ann_json["figures"]:
            obj_id = figure["objectId"]
            video_obj = [obj for obj in video_ann_json["objects"] if obj["id"] == obj_id][0]
            sly_id = video_obj["classId"]
            if sly_id in self.class_map.sly_ids and figure["geometryType"] in allowed_geometries:
                filtered_figures.append(figure)
                # Multiple figures on the same frame can reference the same
                # VideoObject (e.g. when auto-track and predict-video both
                # upload to N+1). Dedupe so VideoObjectCollection.from_json's
                # underlying bidict doesn't raise ValueDuplicationError.
                if obj_id not in seen_obj_ids:
                    seen_obj_ids.add(obj_id)
                    filtered_objects.append(video_obj)

        frame_ann_json = {
            "index": frame_index,
            "figures": filtered_figures,
        }

        return filtered_objects, frame_ann_json

    def after_train_step(self, loss: float):
        self.iter += 1
        self._loss = loss
        if self._should_pause_after_continue:
            self._is_paused = True
            logger.info("Training was paused. Waiting for 1 new sample before resuming...")
            self._wait_until_samples_added(samples_needed=1, max_wait_time=None)
            self._should_pause_after_continue = False
            logger.info("New sample added. Resuming training...")
            self._is_paused = False
        if self.loss_plateau_detector is not None:
            is_plateau = self.loss_plateau_detector.step(loss, self.iter)
            if is_plateau:
                self._is_paused = True
                self._wait_until_samples_added(
                    samples_needed=1,
                    max_wait_time=None,
                )
                self._is_paused = False
                self.loss_plateau_detector.reset()
                logger.debug(f"Resuming training. Next iteration will be {self.iter + 1}")
        self._process_pending_requests()

    def register_model(self, model: nn.Module):
        self.model = model

    def register_dataset(self, dataset: IncrementalDataset):
        assert hasattr(
            dataset, "add_or_update"
        ), "Dataset must implement add_or_update method. Consider inheriting from IncrementalDataset."
        self.dataset = dataset

    def _load_checkpoint(self) -> tuple:
        """Resolve and configure checkpoint based on checkpoint_mode."""
        self._process_pending_requests()
        checkpoint_path, class_map, state = resolve_checkpoint(
            checkpoint_mode=self.checkpoint_mode,
            selected_experiment_task_id=self.selected_experiment_task_id,
            class_map=self.class_map,
            project_meta=self.project_meta,
            api=self.api,
            team_id=self.team_id,
            work_dir=self.work_dir,
        )

        self.class_map = class_map
        self._process_pending_requests()
        return checkpoint_path, state

    def state(self):
        state = {
            "phase": self.phase,
            "iter": self.iter,
            "loss": self._loss,
            "clases": [cls.name for cls in self.class_map.obj_classes],
            "image_ids": self.dataset.get_image_ids() if self.dataset else [],
            "dataset_size": len(self.dataset) if self.dataset else 0,
            "is_paused": self._is_paused,
        }
        return state

    def load_state(self, state: dict):
        self.phase = state.get("phase", Phase.READY_TO_START)
        self.iter = state.get("iter", 0)
        self._loss = state.get("loss", None)
        self.image_ids = state.get("image_ids", [])
        if state.get("is_paused", False):
            self._should_pause_after_continue = True
        dataset_size = state.get("dataset_size", 0)

    def _restore_dataset(self, image_ids: list):
        if not image_ids:
            return

        if isinstance(image_ids[0], int):
            dataset_type = "images"
        elif isinstance(image_ids[0], str):
            dataset_type = "videos"

        if dataset_type == "images":
            logger.info(f"Restoring {len(image_ids)} images from Supervisely...")

            restored_count = 0
            for img_id in image_ids:
                img_info = self.api.image.get_info_by_id(img_id)

                if img_info is None:
                    logger.warning(f"Image {img_id} not found, skipping")
                    continue

                image_np = self.api.image.download_np(img_id)
                ann_json = self.api.annotation.download_json(img_id)
                ann = sly.Annotation.from_json(ann_json, self.project_meta)

                self.dataset.add_or_update(
                    image_id=img_id, image_np=image_np, annotation=ann, image_name=img_info.name
                )

                restored_count += 1

                if restored_count % 10 == 0:
                    logger.info(f"Restored {restored_count}/{len(image_ids)}")

            logger.info(f"Restored {restored_count} images")

        elif dataset_type == "videos":
            logger.info(f"Restoring {len(image_ids)} video frames from Supervisely...")

            restored_count = 0
            for frame_id in image_ids:
                video_id, frame_idx = list(map(int, frame_id.split("_")))
                frame_np = self.api.video.frame.download_np(video_id, frame_idx)

                if restored_count == 0:
                    video_ann_json = self.api.video.annotation.download(video_id)

                video_objects_json, frame_ann_json = self._filter_annotation_video(
                    video_ann_json, frame_idx
                )
                key_id_map = sly.KeyIdMap()
                video_obj_col = sly.VideoObjectCollection.from_json(
                    video_objects_json, self.project_meta, key_id_map
                )
                frames_count = video_ann_json["framesCount"]
                frame_ann = sly.Frame.from_json(
                    frame_ann_json, video_obj_col, frames_count, key_id_map
                )
                frame_h, frame_w = video_ann_json["size"]["height"], video_ann_json["size"]["width"]
                img_ann = self.frame_ann_to_img_ann(frame_ann, frame_h, frame_w)
                self.dataset.add_or_update_video(frame_id, frame_np, img_ann)

                restored_count += 1

                if restored_count % 10 == 0:
                    logger.info(f"Restored {restored_count}/{len(image_ids)}")

            logger.info(f"Restored {restored_count} video frames")

    def prepare_artifacts(self) -> dict:
        """
        Prepare all artifacts for upload (framework-specific).

        Returns:
            Dict with:
                - checkpoint_path: path to checkpoint file
                - checkpoint_info: dict with {name, iteration, loss}
                - config_path: path to config file
                - logs_dir: path to logs directory or None
                - model_name: model name
                - model_config: model configuration dict
                - loss_history: dict with loss history
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement prepare_artifacts()")

    def _get_session_info(self) -> dict:
        """Collect training session context"""
        return {
            "team_id": self.team_id,
            "task_id": self.task_id,
            "project_id": self.project_id,
            "framework_name": self.framework_name,
            "task_type": self.task_type,
            "class_map": self.class_map,
            "start_time": self.training_start_time,
            "train_size": len(self.dataset) if self.dataset else 0,
            "initial_samples": self.initial_samples,
        }

    def _upload_artifacts(self):
        if self._upload_in_progress:
            return

        self._upload_in_progress = True

        try:
            session_info = self._get_session_info()
            artifacts = self.prepare_artifacts()

            report_url = upload_artifacts(
                api=self.api, session_info=session_info, artifacts=artifacts
            )

            logger.info(f"Report: {report_url}")

        except Exception as e:
            logger.error(f"Upload failed: {e}", exc_info=True)

        finally:
            self._upload_in_progress = False

    def _save_and_upload(self):
        """Save checkpoint, state, and upload artifacts"""
        logger.info("Saving checkpoint and uploading artifacts...")
        self.save_checkpoint(self.latest_checkpoint_path)
        save_state_json(self.state(), self.latest_checkpoint_path)
        self._upload_artifacts()

    def save_checkpoint(self, checkpoint_path: str):
        pass

    def _init_loss_plateau_detector(self):
        loss_plateau_detector = LossPlateauDetector(
            window_size=150,
            threshold=0.002,
            patience=3,
            check_interval=1,
        )
        loss_plateau_detector.register_save_checkpoint_callback(self.save_checkpoint)
        return loss_plateau_detector

    def init_evaluator(self):
        return LiveEvaluator(
            task_type=self.task_type,
            class2idx=self.class_map.class2idx,
            ema_alpha=0.1,
            ignore_index=255,
            score_thr=0.3,
        )

    def _release_gpu(self):
        """Drop the training model and any GPU-resident helpers, then empty
        the CUDA cache. Called during shutdown so the GPU is freed during
        the (slow) artifact upload rather than held until process exit.
        """
        try:
            if self.model is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
                del self.model
                self.model = None
            if getattr(self, "_predict_video_tracker", None) is not None:
                del self._predict_video_tracker
                self._predict_video_tracker = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"torch.cuda cleanup failed: {e}")
            logger.info("Released model and emptied CUDA cache")
        except Exception as e:
            logger.warning(f"_release_gpu failed: {e}")

    def _add_shutdown_callback(self):
        """Setup graceful shutdown: save experiment on SIGINT/SIGTERM"""
        self._upload_in_progress = False

        def signal_handler(signum, frame):
            if self._upload_in_progress:
                # Already uploading - force exit on second signal
                signal.signal(signal.SIGINT, lambda s, f: os._exit(1))
                signal.signal(signal.SIGTERM, lambda s, f: os._exit(1))
                return

            # Save checkpoint and state before upload
            logger.info("Received shutdown signal, saving checkpoint...")
            self._process_requests_while_finishing("Training was stopped by user.")
            try:
                logger.info("Saving checkpoint and uploading artifacts...")
                self.save_checkpoint(self.latest_checkpoint_path)
                save_state_json(self.state(), self.latest_checkpoint_path)
            finally:
                # Free the GPU as soon as the checkpoint is on disk so the
                # ~minute-long artifact upload doesn't hold it.
                self._release_gpu()
            try:
                self._upload_artifacts()
            except Exception as e:
                logger.warning(f"artifact upload failed during shutdown: {e}")
            # os._exit bypasses Python's interpreter shutdown — non-daemon
            # threads (mmengine data loaders, CUDA internals) won't be able
            # to keep the process alive past this point.
            os._exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _is_timeout_reached(self, last_time: float, timeout: int) -> bool:
        """Check if timeout interval has passed since last_time"""
        if last_time is None:
            return False
        if timeout <= 0:
            return False
        elapsed = time.time() - last_time
        return elapsed >= timeout

    def _should_upload_periodically(self) -> bool:
        """Check if periodic upload should be triggered"""
        return self._is_timeout_reached(self._last_upload_time, self._upload_interval)

    def _should_stop(self) -> bool:
        """Check if training should stop due to user inactivity"""
        return self._is_timeout_reached(self._last_activity_time, self._inactivity_timeout)

    def _process_requests_while_initializing(self):
        def process_in_background(request: Request):
            try:
                if request.type == RequestType.PREDICT or request.type == RequestType.PREDICT_BATCH:
                    set_exception(
                        request.future,
                        RuntimeError("Cannot run predict, the model is not ready yet."),
                    )
                elif request.type == RequestType.ADD_SAMPLE:
                    result = self._handle_add_sample(request.data)
                    set_result(request.future, result)
                elif request.type == RequestType.ADD_SAMPLE_VIDEO:
                    result = self._handle_add_sample_video(request.data)
                    set_result(request.future, result)
                elif request.type == RequestType.ADD_SAMPLES_VIDEO:
                    result = self._handle_add_samples_video(request.data)
                    set_result(request.future, result)
                elif request.type == RequestType.KEY_FRAMES:
                    # Embedder is independent of the trained model, so we can
                    # serve key-frame selection even before training starts.
                    result = self._handle_key_frames(request.data)
                    set_result(request.future, result)
            except Exception as e:
                logger.error(f"Error processing request {request.type}: {e}", exc_info=True)
                set_exception(request.future, e)

        self._background_request_handler = BackgroundRequestHandler(
            self.request_queue, process_in_background, thread_name="RequestHandlerInitializing"
        )
        self._background_request_handler.start()

    def _process_requests_while_finishing(self, response_message: str):
        def process_in_background(request: Request):
            e = TrainingStoppedException(response_message)
            set_exception(request.future, e)

        self._background_request_handler = BackgroundRequestHandler(
            self.request_queue, process_in_background, thread_name="RequestHandlerFinishing"
        )
        self._background_request_handler.start()

    def _stop_background_request_processing(self):
        if self._background_request_handler is not None:
            self._background_request_handler.stop()
            self._background_request_handler = None
