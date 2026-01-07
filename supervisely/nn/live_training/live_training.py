import os
import numpy as np
from torch import nn
from .api_server import start_api_server
from .request_queue import RequestQueue, RequestType
from .incremental_dataset import IncrementalDataset
from .helpers import ClassMap
import supervisely as sly
from supervisely import logger
from supervisely.nn import TaskType

# Import resolve_checkpoint
from .checkpoint_utils import resolve_checkpoint


class Phase:
    READY_TO_START = "ready_to_start"
    WAITING_FOR_SAMPLES = "waiting_for_samples"
    INITIAL_TRAINING = "initial_training"
    TRAINING = "training"


class LiveTraining:
    
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
        self.initial_samples = initial_samples
        self.filter_classes_by_task = filter_classes_by_task
        if self.task_type is None and self.filter_classes_by_task:
            raise ValueError("task_type must be set in subclass if filter_classes_by_task is set to True")
        
        self.project_id = sly.env.project_id()
        self.team_id = sly.env.team_id()
        self.task_id = sly.env.task_id(raise_not_found=False)
        self.app = sly.Application()
        self.api = sly.Api()
        self.request_queue = RequestQueue()

        if os.getenv("DEVELOP_AND_DEBUG") and not sly.is_production():
            logger.info("🔧 Initializing Develop & Debug application...")
            sly.app.development.supervisely_vpn_network(action="up")
            debug_task = sly.app.development.create_debug_task(self.team_id, port="8000", project_id=self.project_id)

        self._api_thread = start_api_server(self.app, self.request_queue)
        self.phase = Phase.READY_TO_START
        self.iter = 0
        self._loss = None
        self._is_paused = False
        self._ready_to_predict = False
        self.initial_iters = 60  # TODO: remove later
        self.project_meta = self._fetch_project_meta(self.project_id)
        self.class_map = self._init_class_map(self.project_meta)
        self.dataset: IncrementalDataset = None
        self.model: nn.Module  = None
        
        self.checkpoint_mode = os.getenv("modal.state.checkpointMode", "scratch")
        selected_task_id_env = os.getenv("modal.state.selectedExperimentTaskId")
        self.selected_experiment_task_id = int(selected_task_id_env) if selected_task_id_env else None
        self.work_dir = 'app_data'
        self.checkpoint_path = None
        self.dataset_metadata = None
        self.images_ids = []    
    
        # from . import live_training_instance
        # live_training_instance = self  # for access from other modules
    
    def status(self):
        return {
            'phase': self.phase,
            'samples_count': len(self.dataset) if self.dataset is not None else 0,
            'waiting_samples': self.initial_samples,
            'task_type': self.task_type,
            'iteration': self.iter,
            'loss': self._loss,
            'training_paused': self._is_paused,
            'ready_to_predict': self._ready_to_predict,  # TODO: can be removed? phase is used instead (ask Umar)
            'initial_iters': self.initial_iters,
        }
    
    def run(self):
        if checkpoint_mode == "scratch":
            self._run_from_scratch()
        else:
            self._run_from_checkpoint()
        # Phase 1: wait for /start request
        self.phase = Phase.READY_TO_START
        self._resolve_checkpoint_mode()
        self._wait_for_start()

        # Phase 2: add initial samples or restore dataset
        if self.checkpoint_mode == "continue" and self.images_ids:
            logger.info("Restoring dataset from checkpoint...")
            self._restore_dataset()
        else:
            self.phase = Phase.WAITING_FOR_SAMPLES
            self._wait_for_initial_samples()

        # Phase 3: training
        self.phase = Phase.INITIAL_TRAINING
        self.train()

        # TODO: implement uploading weights to Team Files, generate experiment report, etc.

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

    def train(self):
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
        return {
            'image_id': data['image_id'],
            'status': self.status(),
        }
    
    def _fetch_project_meta(self, project_id: int) -> sly.ProjectMeta:
        project_meta = self.api.project.get_meta(project_id)
        project_meta = sly.ProjectMeta.from_json(project_meta)
        return project_meta
    
    def _init_class_map(self, project_meta: sly.ProjectMeta) -> ClassMap:
        # Filter classes according to task_type
        obj_classes = project_meta.obj_classes
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
        self._process_pending_requests()
    
    def register_model(self, model: nn.Module):
        self.model = model
    
    def register_dataset(self, dataset: IncrementalDataset):
        assert hasattr(dataset, 'add_or_update'), "Dataset must implement add_or_update method. Consider inheriting from IncrementalDataset."
        self.dataset = dataset
    

    def _resolve_checkpoint_mode(self):
        """Resolve and configure checkpoint based on checkpoint_mode."""
        self.checkpoint_path, self.class_map, state = resolve_checkpoint(
            checkpoint_mode=self.checkpoint_mode,
            selected_experiment_task_id=self.selected_experiment_task_id,
            class_map=self.class_map,
            project_meta=self.project_meta,
            api=self.api,
            team_id=self.team_id,
            framework_name=self.framework_name,
            work_dir=self.work_dir)
        
        if state is not None:
            self.load_state(state)

    def state(self):
        state = {
            'phase': self.phase,
            'iter': self.iter,
            'loss': self._loss,
            'clases': [cls.name for cls in self.class_map.obj_classes],
            'dataset_metadata': self.dataset_metadata,
            'images_ids': self.dataset.get_images_ids() if self.dataset else [],
            'dataset_size': len(self.dataset) if self.dataset else 0,
            # add more variables as needed
        }
        return state

    def load_state(self, state: dict):
        self.phase = state.get('phase', Phase.READY_TO_START)
        self.iter = state.get('iter', 0)
        self._loss = state.get('loss', None)
        # classes are handled during checkpoint loading
        self.dataset_metadata = state.get('dataset_metadata', {})
        self.images_ids = state.get('images_ids', [])
        dataset_size = state.get('dataset_size', 0)

    def _restore_dataset(self):
        """
        Restore dataset from images_ids by downloading from Supervisely.
        Must be implemented in subclass (framework-specific logic).
        """
        raise NotImplementedError(
            "Subclass must implement _restore_dataset() to download images "
            "from Supervisely and populate IncrementalDataset"
        )