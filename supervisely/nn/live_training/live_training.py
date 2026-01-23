import os
import numpy as np
from .api_server import start_api_server
from .request_queue import RequestQueue, RequestType
from .incremental_dataset import IncrementalDataset
from .helpers import ClassMap
import supervisely as sly
from supervisely import logger
from supervisely.nn import TaskType
from datetime import datetime
import signal
import sys
import time
from .checkpoint_utils import resolve_checkpoint, save_state_json
from .artifacts_utils import upload_artifacts
from .loss_plateau_detector import LossPlateauDetector
from pathlib import Path

class Phase:
    READY_TO_START = "ready_to_start"
    WAITING_FOR_SAMPLES = "waiting_for_samples"
    INITIAL_TRAINING = "initial_training"
    TRAINING = "training"

class LiveTraining:
    
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
        self._api_thread = start_api_server(self.app, self.request_queue)
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
        }
    
    def run(self):
        self.training_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._add_shutdown_callback()

        work_dir_path = Path(self.work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)
        model_meta_path = work_dir_path / "model_meta.json"
        sly.json.dump_json_file(self.project_meta.to_json(), str(model_meta_path))
        
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
                final_checkpoint = self.latest_checkpoint_path
                self.save_checkpoint(final_checkpoint)
                save_state_json(self.state(), final_checkpoint)
                self._upload_artifacts()
        
    def _run_from_scratch(self):
        self.phase = Phase.WAITING_FOR_SAMPLES
        self._wait_for_initial_samples()
        self.train(checkpoint_path=None)
    
    def _run_continue(self):
        checkpoint_path, state = self._load_checkpoint()
        self.load_state(state)
        image_ids = state.get('image_ids', [])
        if image_ids:
            self._restore_dataset(image_ids)
        self.train(checkpoint_path=checkpoint_path)
    
    def _run_finetune(self):
        checkpoint_path, _ = self._load_checkpoint()
        self.phase = Phase.WAITING_FOR_SAMPLES
        self._wait_for_initial_samples()
        self.train(checkpoint_path=checkpoint_path)
    
    def _wait_for_start(self):
        request = self.request_queue.get()
        while request.type != RequestType.START:
            if request.type == RequestType.STATUS:
                status = self.status()
                request.future.set_result(status)
            else:
                request.future.set_exception(Exception(f"Unexpected request {request.type} while waiting for START"))
            request = self.request_queue.get()
        # When START is received
        status = self.status()
        status['phase'] = Phase.WAITING_FOR_SAMPLES
        request.future.set_result(status)

    def _wait_until_samples_added(
        self,
        samples_needed: int,
        max_wait_time: int = None,
    ):
        sleep_interval = 0.5
        elapsed_time = 0
        samples_before = len(self.dataset)

        while len(self.dataset) - samples_before < samples_needed:
            if max_wait_time is not None and elapsed_time >= max_wait_time:
                raise RuntimeError("Timeout waiting for samples")
            
            if not self.request_queue.is_empty():
                self._process_pending_requests()

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
        requests = self.request_queue.get_all()
        if not requests:
            return

        new_samples_added = False
        
        for request in requests: 
            try:
                if request.type == RequestType.PREDICT:
                    result = self._handle_predict(request.data)
                    request.future.set_result(result)
                
                elif request.type == RequestType.ADD_SAMPLE:
                    result = self._handle_add_sample(request.data)
                    request.future.set_result(result)
                    new_samples_added = True

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
        """
        raise NotImplementedError

    def _handle_predict(self, data: dict):
        image_np = data['image']
        image_info = {'id': data['image_id']}
        model = self.model
        was_training = model.training
        model.eval()
        try:
            objects = self.predict(self.model, image_np=image_np, image_info=image_info)
            return {
                'objects': objects,
                'image_id': data['image_id'],
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
        return self.dataset.add_or_update(image_id, image_np, annotation, image_name)
    
    def _handle_add_sample(self, data: dict):
        ann_json = data['annotation']
        ann_json = self._filter_annotation(ann_json)
        sly_ann = sly.Annotation.from_json(ann_json, self.project_meta)
        self.add_sample(
            image_id=data['image_id'],
            image_np=data['image'],
            annotation=sly_ann,
            image_name=data['image_name']
        )
        if (len(self.dataset) >= self.initial_samples) and self.phase==Phase.WAITING_FOR_SAMPLES:
            self.phase = Phase.INITIAL_TRAINING
        return {
            'image_id': data['image_id'],
            'status': self.status(),
        }
    
    def _fetch_project_meta(self, project_id: int) -> sly.ProjectMeta:
        project_meta = self.api.project.get_meta(project_id)
        project_meta = sly.ProjectMeta.from_json(project_meta)
        return project_meta
    
    def _init_class_map(self, project_meta: sly.ProjectMeta) -> ClassMap:
        obj_classes = list(project_meta.obj_classes)

        if self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            obj_classes.insert(0, sly.ObjClass(name='_background_', geometry_type=sly.Bitmap))

        if self.filter_classes_by_task:
            allowed_geometries = self._task2geometries[self.task_type]
            obj_classes = [
                obj_class for obj_class in obj_classes
                if obj_class.geometry_type in allowed_geometries
            ]

        return ClassMap(obj_classes)
    
    def _filter_annotation(self, ann_json: dict) -> dict:
        # Filter objects according to class_map
        # Important: Must be filtered before sly.Annotation.from_json due to static project meta
        filtered_objects = []
        for obj in ann_json['objects']:
            sly_id = obj['classId']
            if sly_id in self.class_map.sly_ids:
                filtered_objects.append(obj)
        ann_json['objects'] = filtered_objects
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
            'is_paused': self._is_paused
        }
        return state

    def load_state(self, state: dict):
        self.phase = state.get('phase', Phase.READY_TO_START)
        self.iter = state.get('iter', 0)
        self._loss = state.get('loss', None)
        self.image_ids = state.get('image_ids', [])
        if state.get('is_paused', False):
            self._should_pause_after_continue = True
        dataset_size = state.get('dataset_size', 0)

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
    
    def save_checkpoint(self, checkpoint_path: str):
        pass

    def _init_loss_plateau_detector(self):
        loss_plateau_detector = LossPlateauDetector()
        loss_plateau_detector.register_save_checkpoint_callback(self.save_checkpoint)
        return loss_plateau_detector
    
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
            self.save_checkpoint(self.latest_checkpoint_path)
            save_state_json(self.state(), self.latest_checkpoint_path)
            self._upload_artifacts()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
