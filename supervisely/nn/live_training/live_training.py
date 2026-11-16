import os
import numpy as np
from .api_server import start_api_server
from .request_queue import RequestQueue, RequestType, Request
from .incremental_dataset import IncrementalDataset
from .helpers import ClassMap
from .evaluator import LiveEvaluator
import supervisely as sly
from supervisely import logger
from supervisely.nn import TaskType
from datetime import datetime
import signal
import sys
import threading
import time
from .checkpoint_utils import resolve_checkpoint, save_state_json
from .artifacts_utils import upload_artifacts
from .loss_plateau_detector import LossPlateauDetector
from .utils import TrainingStoppedException, BackgroundRequestHandler, set_exception, set_result
from .settings import LiveTrainingSettings, OverlapMode
from pathlib import Path
from typing import List, Optional

class Phase:
    """String constants describing the live-training lifecycle phases."""

    READY_TO_START = "ready_to_start"
    WAITING_FOR_SAMPLES = "waiting_for_samples"
    # Training on existing annotations (seed) chosen before Start
    PRETRAIN_ON_EXISTING_ANNOTATIONS = "pretrain_on_existing_annotations"
    INITIAL_TRAINING = "initial_training"
    TRAINING = "training"
    STOPPING = "stopping"

class LiveTraining:
    """Base implementation of an interactive/live training loop driven by requests (start/add sample/predict/status)."""

    from torch import nn  # pylint: disable=import-error
    task_type: str = None  # Should be set in subclass
    framework_name: str = None # Should be set in subclass

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
            raise ValueError("task_type must be set in subclass if filter_classes_by_task is set to True")
        if self.framework_name is None:
            raise ValueError("framework_name must be set in subclass")

        self.project_id = sly.env.project_id()
        self.team_id = sly.env.team_id()
        self.task_id = sly.env.task_id(raise_not_found=False)
        self.app = sly.Application()
        self.api = sly.Api()
        self.request_queue = RequestQueue()

        if os.getenv("DEVELOP_AND_DEBUG") and not sly.is_production():
            logger.info(f"🔧 Initializing Develop & Debug application for project {self.project_id}...")
            sly.app.development.supervisely_vpn_network(action="up")
            debug_task = sly.app.development.create_debug_task(self.team_id, port="8000", project_id=self.project_id)
            self.task_id = debug_task['id']
        self._api_thread = start_api_server(self.app, self.request_queue, self)
        self.phase = Phase.READY_TO_START
        self.iter = 0
        self._loss = None
        self._is_paused = False
        self._should_pause_after_continue = False
        self.initial_iters = 60  # TODO: remove later
        self.project_meta = self._fetch_project_meta(self.project_id)
        self.class_map = self._init_class_map(self.project_meta)
        self.dataset: IncrementalDataset = None
        self.model: nn.Module  = None
        self.loss_plateau_detector = self._init_loss_plateau_detector()
        self.work_dir = 'app_data'
        self.latest_checkpoint_path = f"{self.work_dir}/checkpoints/latest.pth"

        self.checkpoint_mode = os.getenv("modal.state.checkpointMode", "scratch")
        selected_task_id_env = os.getenv("modal.state.selectedExperimentTaskId")
        self.selected_experiment_task_id = int(selected_task_id_env) if selected_task_id_env else None

        self.training_start_time = None
        self._upload_in_progress = False

        self.evaluator = self.init_evaluator()
        self._upload_interval = 7200
        self._last_upload_time = None

        # /status normally bypasses the request queue and returns immediately.
        # In "continue" mode that would let the UI poll status during the
        # (possibly slow) checkpoint restore and observe inconsistent state.
        # _start_received marks the moment START was processed; until
        # _continue_checkpoint_loaded is set, /status held after START will
        # block. Status requests received before START always pass through.
        self._start_received = threading.Event()
        self._continue_checkpoint_loaded = threading.Event()

        self._inactivity_timeout = 24 * 3600 # 24 hours in seconds
        self._last_activity_time = None
        self._background_request_handler: BackgroundRequestHandler = None

        # Session setup received with START (classes, overlap rule, seed). Fixed for the session.
        self.settings: Optional[LiveTrainingSettings] = None
        # Unselected classes painted as ignore (255) when settings.ignore_unselected is set
        self._ignore_class_ids: set = set()
        # Seed: existing annotations to train on before live labeling
        self._seed_images: List[sly.ImageInfo] = []
        self._seed_total = 0
        self._seed_loaded_ids: List[int] = []
        self._seed_failed = 0
        self._seed_loading_done = threading.Event()
        self._seed_thread: threading.Thread = None
        # Samples may be added from the seed loader thread and from request handlers
        self._dataset_lock = threading.Lock()

        # from . import live_training_instance
        # live_training_instance = self  # for access from other modules

    @property
    def ready_to_predict(self):
        return self.iter > self.initial_iters

    def status(self):
        return {
            'phase': self.phase,
            'samples_count': len(self.dataset) if self.dataset is not None else 0,
            'waiting_samples': self.initial_samples,
            'task_type': self.task_type,
            'iteration': self.iter,
            'loss': self._loss,
            'training_paused': self._is_paused,
            'ready_to_predict': self.ready_to_predict,
            'initial_iters': self.initial_iters,
            'model_quality': self.evaluator.ema_value if self.evaluator else None,
            'seed_progress': self._seed_progress(),
        }

    def get_settings(self) -> Optional[dict]:
        """Session settings in the /start payload format, or None before START."""
        return self.settings.to_json() if self.settings is not None else None

    def run(self):
        self.training_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._add_shutdown_callback()
        self._last_upload_time = time.time()
        self._last_activity_time = time.time()

        Path(self.work_dir).mkdir(parents=True, exist_ok=True)

        try:
            self.phase = Phase.READY_TO_START
            self._wait_for_start()
            if self.checkpoint_mode == 'continue':
                self._run_continue()
            elif self.checkpoint_mode == 'finetune':
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
        self._prepare_initial_samples()
        self._process_requests_while_initializing()
        self.train(checkpoint_path=None)

    def _run_continue(self):
        try:
            checkpoint_path, state = self._load_checkpoint()
            self.load_state(state)
            self._apply_settings()
            image_ids = state.get("image_ids", [])
            if image_ids:
                # Seed images of the previous session are among image_ids: no new seeding
                self._restore_dataset(image_ids)
        finally:
            # Release any /status requests blocked behind the checkpoint
            # restore even if loading failed — the outer try/except in
            # run() handles the exception, and a hung /status is worse
            # than a status reporting partial state.
            self._continue_checkpoint_loaded.set()
        self._process_requests_while_initializing()
        self.train(checkpoint_path=checkpoint_path)

    def _run_finetune(self):
        checkpoint_path, _ = self._load_checkpoint()
        self._prepare_initial_samples()
        self._process_requests_while_initializing()
        self.train(checkpoint_path=checkpoint_path)

    def _prepare_initial_samples(self):
        """Start seeding from existing annotations (if any) and wait for enough samples to start training."""
        if self._seed_total > 0:
            self.phase = Phase.PRETRAIN_ON_EXISTING_ANNOTATIONS
            self._start_seed_loader()
            self._wait_for_seed_samples()
            if self.phase == Phase.PRETRAIN_ON_EXISTING_ANNOTATIONS:
                return
        self.phase = Phase.WAITING_FOR_SAMPLES
        self._wait_for_initial_samples()

    def _wait_for_start(self):
        request = self.request_queue.get()
        while request.type != RequestType.START:
            if request.type == RequestType.STATUS:
                status = self.status()
                request.future.set_result(status)
            else:
                request.future.set_exception(Exception(f"Unexpected request {request.type} while waiting for START"))
            request = self.request_queue.get()
        # When START is received — refresh meta in case classes were added after app launch
        self.project_meta = self._fetch_project_meta(self.project_id)
        try:
            self._setup_session(request.data)
        except ValueError as e:
            # Deliver the error as the direct response to the START request,
            # then re-raise so the session fails loud instead of training
            # with invalid settings (e.g. a model with num_classes=0).
            request.future.set_exception(e)
            raise
        status = self.status()
        if self._seed_total > 0:
            status['phase'] = Phase.PRETRAIN_ON_EXISTING_ANNOTATIONS
        else:
            status['phase'] = Phase.WAITING_FOR_SAMPLES
        # Flip _start_received before resolving the START future so any
        # /status the client sends after seeing the START response is
        # guaranteed to observe the flag set.
        self._start_received.set()
        request.future.set_result(status)

    def _setup_session(self, start_data: Optional[dict]):
        """
        Apply the session setup received with START: classes to train, overlap rule
        and seed (existing annotations). Without settings in the payload all eligible
        classes are trained, as before.

        :raises ValueError: if the settings are invalid.
        """
        eligible_classes = self._get_eligible_classes(self.project_meta)
        settings = None
        if self.checkpoint_mode == 'continue':
            # Setup is restored from the checkpoint in _run_continue (load_state)
            if start_data and start_data.get("classes") is not None:
                logger.info("Continue mode: session settings are restored from the checkpoint, /start settings are ignored")
        else:
            settings = LiveTrainingSettings.from_json(start_data)
        if settings is None:
            settings = LiveTrainingSettings.default(eligible_classes)
        else:
            settings.validate(self.project_meta, eligible_classes)
        self.settings = settings
        logger.info(f"Live training settings: {settings.to_json()}")

        self.class_map = self._init_class_map(self.project_meta, selected_ids=settings.classes)
        self._validate_class_map()
        self._apply_settings()

        if settings.has_seed:
            self._seed_images = self._resolve_seed_images(settings)
            self._seed_total = len(self._seed_images)
            logger.info(f"Existing annotations: {self._seed_total} images will be used for training")

    def _apply_settings(self):
        """Propagate class map and rasterization settings to the dataset and evaluator."""
        settings = self.settings
        selected_ids = set(self.class_map.sly_ids)
        id2class = {obj_class.sly_id: obj_class for obj_class in self.project_meta.obj_classes}

        self._ignore_class_ids = set()
        if settings.ignore_unselected and self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            self._ignore_class_ids = {
                obj_class.sly_id
                for obj_class in self._get_eligible_classes(self.project_meta)
                if obj_class.sly_id not in selected_ids
            }
        ignore_class_names = [id2class[class_id].name for class_id in self._ignore_class_ids]

        class_priority = None
        if settings.overlap == OverlapMode.CLASS_PRIORITY:
            class_priority = [id2class[c].name for c in settings.class_order if c in id2class]

        if self.dataset is not None:
            self.dataset.update_class_map(self.class_map.class2idx)
            if hasattr(self.dataset, "set_rasterization"):
                self.dataset.set_rasterization(ignore_class_names, class_priority)
        if self.evaluator is not None:
            self.evaluator.class2idx = self.class_map.class2idx
            self.evaluator.ignore_class_names = set(ignore_class_names)

        if ignore_class_names:
            logger.info(f"Unselected classes are ignored during training: {ignore_class_names}")
        if class_priority:
            logger.info(f"Overlap rule: class priority (top first): {class_priority}")

        work_dir_path = Path(self.work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)
        model_meta = sly.ProjectMeta(obj_classes=self.class_map.obj_classes)
        sly.json.dump_json_file(model_meta.to_json(), str(work_dir_path / "model_meta.json"))

    def _seed_progress(self) -> Optional[dict]:
        if self._seed_total == 0:
            return None
        if self.dataset is None or self.phase != Phase.PRETRAIN_ON_EXISTING_ANNOTATIONS:
            # Seeding finished (or the session continues from a checkpoint)
            return {'done': self._seed_total, 'total': self._seed_total}
        sampled = self.dataset.sampled_image_ids
        done = sum(1 for image_id in self._seed_loaded_ids if image_id in sampled)
        total = self._seed_total - self._seed_failed
        return {'done': min(done, total), 'total': total}

    def _resolve_seed_images(self, settings: LiveTrainingSettings) -> List[sly.ImageInfo]:
        """Resolve seed dataset ids + image filters into a list of images (nested datasets included)."""
        dataset_ids = []
        for dataset_id in settings.seed_dataset_ids:
            dataset_info = self.api.dataset.get_info_by_id(dataset_id)
            if dataset_info is None:
                raise ValueError(f"Dataset with id {dataset_id} not found.")
            if dataset_info.project_id != self.project_id:
                raise ValueError(
                    f"Dataset '{dataset_info.name}' (id {dataset_id}) does not belong to the project {self.project_id}."
                )
            dataset_ids.append(dataset_id)
            nested = self.api.dataset.get_nested(self.project_id, dataset_id)
            dataset_ids.extend(ds.id for ds in nested)
        dataset_ids = list(dict.fromkeys(dataset_ids))

        images = {}
        for dataset_id in dataset_ids:
            try:
                infos = self.api.image.get_filtered_list(
                    dataset_id=dataset_id, filters=settings.seed_filters or None
                )
            except Exception as e:
                raise ValueError(f"Failed to list images of dataset {dataset_id} with the given filters: {e}") from e
            for info in infos:
                images.setdefault(info.id, info)
        return list(images.values())

    def _start_seed_loader(self):
        self._seed_loading_done.clear()
        self._seed_thread = threading.Thread(target=self._load_seed_images, daemon=True, name="SeedLoader")
        self._seed_thread.start()

    def _load_seed_images(self, batch_size: int = 16):
        """Download seed images with annotations and add them to the training set (runs in a thread)."""
        by_dataset = {}
        for info in self._seed_images:
            by_dataset.setdefault(info.dataset_id, []).append(info)
        try:
            for dataset_id, infos in by_dataset.items():
                for batch in sly.batched(infos, batch_size):
                    ids = [info.id for info in batch]
                    try:
                        images_np = self.api.image.download_nps(dataset_id, ids)
                        ann_jsons = self.api.annotation.download_json_batch(dataset_id, ids)
                    except Exception as e:
                        logger.error(f"Failed to download {len(ids)} seed images: {e}", exc_info=True)
                        self._seed_failed += len(ids)
                        continue
                    for info, image_np, ann_json in zip(batch, images_np, ann_jsons):
                        try:
                            ann_json = self._filter_annotation(ann_json)
                            ann = sly.Annotation.from_json(ann_json, self.project_meta)
                            self.add_sample(
                                image_id=info.id,
                                image_np=image_np,
                                annotation=ann,
                                image_name=info.name,
                            )
                            self._seed_loaded_ids.append(info.id)
                        except Exception as e:
                            logger.error(f"Failed to add seed image {info.id}: {e}", exc_info=True)
                            self._seed_failed += 1
                    logger.info(
                        f"Existing annotations: {len(self._seed_loaded_ids)}/{self._seed_total} images loaded"
                    )
        finally:
            self._seed_loading_done.set()
        if self._seed_failed:
            logger.warning(f"Existing annotations: {self._seed_failed} images failed to load and were skipped")

    def _wait_for_seed_samples(self):
        """Wait until the seed gives enough samples to start training. Requests are served meanwhile."""
        target = min(self.initial_samples, self._seed_total)
        while len(self.dataset) < target and not self._seed_loading_done.is_set():
            if not self.request_queue.is_empty():
                self._process_pending_requests()
            time.sleep(0.3)
        if len(self.dataset) < self.initial_samples and self._seed_loading_done.is_set():
            # Seed is too small to start: wait for samples from the labeling tool as usual
            logger.info(
                f"Existing annotations gave {len(self.dataset)} samples, "
                f"{self.initial_samples} are needed to start training"
            )
            self.phase = Phase.WAITING_FOR_SAMPLES

    def _check_seed_finished(self):
        """Switch from the seed phase to live training once every seed image was trained on."""
        if self.phase != Phase.PRETRAIN_ON_EXISTING_ANNOTATIONS:
            return
        if not self._seed_loading_done.is_set():
            return
        sampled = self.dataset.sampled_image_ids
        if all(image_id in sampled for image_id in self._seed_loaded_ids):
            # The model is already trained on existing annotations: no initial training phase
            self.phase = Phase.TRAINING
            logger.info(
                f"Trained on all {len(self._seed_loaded_ids)} existing images. Switching to live training"
            )

    def _wait_until_samples_added(
        self,
        samples_needed: int,
        max_wait_time: int = None,
    ):
        sleep_interval = 0.3
        elapsed_time = 0
        samples_before = len(self.dataset)

        while len(self.dataset) - samples_before < samples_needed:
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
        if len(self.dataset) >= self.initial_samples:
            return

        self.phase = Phase.WAITING_FOR_SAMPLES
        self._is_paused = True

        samples_needed = self.initial_samples - len(self.dataset)
        logger.info(f"Waiting for {samples_needed} initial samples")
        self._wait_until_samples_added(
            samples_needed=samples_needed,
            max_wait_time=3600,
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

                elif request.type == RequestType.ADD_SAMPLE:
                    result = self._handle_add_sample(request.data)
                    request.future.set_result(result)
                    self._last_activity_time = time.time()

                elif request.type == RequestType.STATUS:
                    result = self.status()
                    request.future.set_result(result)

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

        image_info: {'id': image id, 'instance_masks': bool} - for segmentation, whether to split
        each class mask into separate objects (True, default) or return one mask per class (False).
        """
        raise NotImplementedError

    def predict_batch(self, model: nn.Module, images_np: list, images_info: list) -> list:
        """
        Run inference on several images. Returns a list with one list of sly figures (json) per image,
        in the same order as images_np.

        Default implementation calls predict() for each image. Override for batched inference.
        """
        return [
            self.predict(model, image_np=image_np, image_info=image_info)
            for image_np, image_info in zip(images_np, images_info)
        ]

    def _handle_predict(self, data: dict):
        images_np = data['images']
        image_ids = data['image_ids']
        instance_masks = data.get('instance_masks', True)
        images_info = [{'id': image_id, 'instance_masks': instance_masks} for image_id in image_ids]
        model = self.model
        was_training = model.training
        model.eval()
        try:
            objects_per_image = self.predict_batch(self.model, images_np=images_np, images_info=images_info)

            objects = []
            for image_id, image_np, objects_raw in zip(image_ids, images_np, objects_per_image):
                if self.evaluator:
                    image_shape = image_np.shape[:2]
                    self.evaluator.store_prediction(image_id, objects_raw, image_shape)
                # Flat list of objects for all images, each object points to its image
                objects.extend({**obj, 'image_id': image_id} for obj in objects_raw)

            return {
                'objects': objects,
                'image_id': image_ids[0],
                'image_ids': image_ids,
                'status': self.status(),
            }
        finally:
            # Restore training mode
            if was_training:
                model.train()

    def add_sample(
            self,
            image_id: int,
            image_np: np.ndarray,
            annotation: sly.Annotation,
            image_name: str
        ) -> dict:
        with self._dataset_lock:
            return self.dataset.add_or_update(image_id, image_np, annotation, image_name)

    def _handle_add_sample(self, data: dict):
        ann_json = data['annotation']
        ann_json = self._filter_annotation(ann_json)
        sly_ann = sly.Annotation.from_json(ann_json, self.project_meta)
        image_id = data['image_id']
        self.add_sample(
            image_id=image_id,
            image_np=data['image'],
            annotation=sly_ann,
            image_name=data['image_name']
        )
        if self.evaluator and self.phase!=Phase.WAITING_FOR_SAMPLES:
            result = self.evaluator.evaluate(image_id, sly_ann)
            if result is not None:
                metric_name = self.evaluator.metric_name
                logger.info(
                    f"Image {image_id}: {metric_name}={result['metric_value']:.3f}, "
                    f"EMA={result['ema_value']:.3f}"
                )

        status_message = f"{len(self.dataset)} images added"
        if self.phase != Phase.WAITING_FOR_SAMPLES:
            quality = self.evaluator.ema_value if self.evaluator else None
            if quality is None or quality < 0.3:
                status_message += " | Learning..."
            else:
                status_message += f" | Model Quality: {quality:.2f}"
        sly.Progress(status_message, 1, log_extra={"hide_counter": True}).iter_done_report()

        if (len(self.dataset) >= self.initial_samples) and self.phase==Phase.WAITING_FOR_SAMPLES:
            self.phase = Phase.INITIAL_TRAINING

        return {
            'image_id': image_id,
            'status': self.status(),
        }

    def _fetch_project_meta(self, project_id: int) -> sly.ProjectMeta:
        project_meta = self.api.project.get_meta(project_id)
        project_meta = sly.ProjectMeta.from_json(project_meta)
        return project_meta

    def _get_eligible_classes(self, project_meta: sly.ProjectMeta) -> List[sly.ObjClass]:
        """Project classes whose geometry fits the task (all classes if filter_classes_by_task is off)."""
        obj_classes = list(project_meta.obj_classes)
        if self.filter_classes_by_task:
            allowed_geometries = self._task2geometries[self.task_type] + [sly.AnyGeometry]
            obj_classes = [
                obj_class for obj_class in obj_classes
                if obj_class.geometry_type in allowed_geometries
            ]
        return obj_classes

    def _init_class_map(self, project_meta: sly.ProjectMeta, selected_ids: List[int] = None) -> ClassMap:
        """
        :param selected_ids: Class ids chosen for training. If None, all eligible classes are used.
        """
        obj_classes = self._get_eligible_classes(project_meta)
        if selected_ids is not None:
            selected_ids = set(selected_ids)
            obj_classes = [obj_class for obj_class in obj_classes if obj_class.sly_id in selected_ids]

        if self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            obj_classes.insert(0, sly.ObjClass(name='_background_', geometry_type=sly.Bitmap))

        return ClassMap(obj_classes)

    def _validate_class_map(self):
        if any(obj_class.name != '_background_' for obj_class in self.class_map.obj_classes):
            return
        if self.filter_classes_by_task and self.task_type is not None:
            allowed_geometries = self._task2geometries[self.task_type]
            geometry_names = [geom.geometry_name() for geom in allowed_geometries]
            raise ValueError(
                f"The project has no classes suitable for {self.task_type}. "
                f"Live training requires at least one class with geometry type "
                f"{', '.join(geometry_names)} (or 'Any Shape' geometry). "
                f"Please add a suitable class to the project and restart the live training session."
            )
        raise ValueError(
            "The project has no classes. Please add at least one class to the project "
            "and restart the live training session."
        )

    def _filter_annotation(self, ann_json: dict) -> dict:
        # Filter objects according to class_map
        # Important: Must be filtered before sly.Annotation.from_json due to static project meta
        allowed_geometries = self._task2geometries[self.task_type]
        allowed_geometries = [geom.geometry_name() for geom in allowed_geometries]
        filtered_objects = []
        dropped_objects = []
        for obj in ann_json['objects']:
            sly_id = obj['classId']
            is_known = sly_id in self.class_map.sly_ids or sly_id in self._ignore_class_ids
            if is_known and obj['geometryType'] in allowed_geometries:
                filtered_objects.append(obj)
            else:
                dropped_objects.append(obj)
        ann_json['objects'] = filtered_objects
        if dropped_objects:
            dropped_info = {
                f"{obj.get('classTitle', obj.get('classId'))} ({obj['geometryType']})"
                for obj in dropped_objects
            }
            logger.warning(
                f"Skipped {len(dropped_objects)} object(s) not matching class_map/allowed geometries: "
                f"{sorted(dropped_info)}. Allowed geometries: {allowed_geometries}"
            )
        return ann_json

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
        self._check_seed_finished()
        # No plateau pause while training on existing annotations: no new samples come in this phase
        if self.loss_plateau_detector is not None and self.phase != Phase.PRETRAIN_ON_EXISTING_ANNOTATIONS:
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
        assert hasattr(dataset, 'add_or_update'), "Dataset must implement add_or_update method. Consider inheriting from IncrementalDataset."
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
            work_dir=self.work_dir
        )

        self.class_map = class_map
        self._validate_class_map()
        self._apply_settings()
        self._process_pending_requests()
        return checkpoint_path, state

    def state(self):
        state = {
            'phase': self.phase,
            'iter': self.iter,
            'loss': self._loss,
            'clases': [cls.name for cls in self.class_map.obj_classes],
            'image_ids': self.dataset.get_image_ids() if self.dataset else [],
            'dataset_size': len(self.dataset) if self.dataset else 0,
            'is_paused': self._is_paused,
            'evaluator': self.evaluator.state_dict() if self.evaluator else None,
            'settings': self.get_settings(),
            'seed_total': self._seed_total,
        }
        return state

    def load_state(self, state: dict):
        self.phase = state.get('phase', Phase.READY_TO_START)
        self.iter = state.get('iter', 0)
        if self.phase == Phase.PRETRAIN_ON_EXISTING_ANNOTATIONS:
            # Seed images loaded so far are restored as regular samples; the seed is not resumed
            self.phase = Phase.TRAINING
        settings = LiveTrainingSettings.from_json(state.get('settings'))
        if settings is None:
            # Checkpoint from a version without session settings: classes come from the checkpoint
            class_ids = [c.sly_id for c in self.class_map.obj_classes if c.sly_id is not None]
            settings = LiveTrainingSettings(classes=class_ids, class_order=list(class_ids))
        self.settings = settings
        self._seed_total = state.get('seed_total', 0)
        self._loss = state.get('loss', None)
        self.image_ids = state.get('image_ids', [])
        if state.get('is_paused', False):
            self._should_pause_after_continue = True
        dataset_size = state.get('dataset_size', 0)
        evaluator_state = state.get('evaluator')
        if self.evaluator and evaluator_state:
            self.evaluator.load_state_dict(evaluator_state)

    def _restore_dataset(self, image_ids: list):
        if not image_ids:
            return

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
                image_id=img_id,
                image_np=image_np,
                annotation=ann,
                image_name=img_info.name
            )

            restored_count += 1

            if restored_count % 10 == 0:
                logger.info(f"Restored {restored_count}/{len(image_ids)}")

        logger.info(f"Restored {restored_count} images")

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
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement prepare_artifacts()"
        )

    def _get_session_info(self) -> dict:
        """Collect training session context"""
        return {
            'team_id': self.team_id,
            'task_id': self.task_id,
            'project_id': self.project_id,
            'framework_name': self.framework_name,
            'task_type': self.task_type,
            'class_map': self.class_map,
            'start_time': self.training_start_time,
            'train_size': len(self.dataset) if self.dataset else 0,
            'initial_samples': self.initial_samples
        }

    def _upload_artifacts(self):
        if self._upload_in_progress:
            return

        self._upload_in_progress = True

        try:
            session_info = self._get_session_info()
            artifacts = self.prepare_artifacts()

            report_url = upload_artifacts(
                api=self.api,
                session_info=session_info,
                artifacts=artifacts
            )

            logger.info(f"Report: {report_url}")

        except Exception as e:
            logger.error(f"Upload failed: {e}", exc_info=True)

        finally:
            self._upload_in_progress = False

    def _save_and_upload(self):
        """Save checkpoint, state, and upload artifacts"""
        if self.iter == 0:
            return
        logger.info("Saving checkpoint and uploading artifacts...")
        try:
            self.save_checkpoint(self.latest_checkpoint_path)
            save_state_json(self.state(), self.latest_checkpoint_path)
            self._upload_artifacts()
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

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

    def _add_shutdown_callback(self):
        """Setup graceful shutdown: save experiment on SIGINT/SIGTERM"""
        self._upload_in_progress = False

        def signal_handler(signum, frame):
            if self._upload_in_progress:
                # Already uploading - force exit on second signal
                signal.signal(signal.SIGINT, lambda s, f: sys.exit(1))
                signal.signal(signal.SIGTERM, lambda s, f: sys.exit(1))
                return

            # Save checkpoint and state before upload
            logger.info("Received shutdown signal, saving checkpoint...")
            self.phase = Phase.STOPPING
            self._process_requests_while_finishing("Training was stopped by user.")
            self._save_and_upload()
            sys.exit(0)

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
                if request.type == RequestType.PREDICT:
                    set_exception(request.future, RuntimeError("Cannot run predict, the model is not ready yet."))
                elif request.type == RequestType.ADD_SAMPLE:
                    result = self._handle_add_sample(request.data)
                    set_result(request.future, result)
                elif request.type == RequestType.STATUS:
                    result = self.status()
                    set_result(request.future, result)
            except Exception as e:
                logger.error(f"Error processing request {request.type}: {e}", exc_info=True)
                set_exception(request.future, e)
        self._background_request_handler = BackgroundRequestHandler(self.request_queue, process_in_background, thread_name="RequestHandlerInitializing")
        self._background_request_handler.start()
    
    def _process_requests_while_finishing(self, response_message: str):
        def process_in_background(request: Request):
            e = TrainingStoppedException(response_message)
            set_exception(request.future, e)
        self._background_request_handler = BackgroundRequestHandler(self.request_queue, process_in_background, thread_name="RequestHandlerFinishing")
        self._background_request_handler.start()
    
    def _stop_background_request_processing(self):
        if self._background_request_handler is not None:
            self._background_request_handler.stop()
            self._background_request_handler = None
