import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import supervisely as sly
from supervisely import logger
from supervisely.template.live_training.live_training_generator import LiveTrainingGenerator
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
import yaml
import re
import torch

class OnlineTrainingArtifactsUploader:
    """Upload artifacts and generate Live Training report"""
    
    def __init__(
        self,
        api: sly.Api,
        project_id: int,
        work_dir: str,
        config_file: str,
        framework_name: str,
        model_name: str,
        task_type: str,
        task_id: Optional[int] = None,
        team_id: Optional[int] = None
    ):
        self.api = api
        self.project_id = project_id
        self.work_dir = Path(work_dir)
        self.config_file = config_file
        self.framework_name = framework_name
        self.model_name = model_name
        self.task_type = task_type
        self.task_id = task_id
        self.team_id = team_id
        # self.session_id = int(datetime.now().timestamp())
        self.output_dir = self.work_dir / "upload_artifacts"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    # ============= CHECKPOINTS =============
    
    def _prepare_checkpoints(self, model_meta: sly.ProjectMeta, model_name: str) -> Tuple[List[str], List[dict]]:
        """Prepare checkpoints with metadata"""
        
        # Parse logs once for all checkpoints
        loss_lookup = self._build_loss_lookup()
        
        checkpoint_paths = []
        checkpoints_info = []
        
        for ckpt_file in sorted(self.work_dir.glob("*.pth")):
            if ckpt_file.name == "latest.pth":
                continue
            
            try:
                paths, info = self._process_checkpoint(
                    ckpt_file, model_meta, model_name, loss_lookup
                )
                checkpoint_paths.append(paths)
                checkpoints_info.append(info)
                logger.info(f"✅ {ckpt_file.name}: iter={info['iteration']}, loss={info['loss']}")
            except Exception as e:
                logger.warning(f"Failed to process {ckpt_file.name}: {e}")
                # Fallback: simple copy
                shutil.copy2(str(ckpt_file), str(self.checkpoints_dir / ckpt_file.name))
                checkpoint_paths.append(f"checkpoints/{ckpt_file.name}")
                checkpoints_info.append({"name": ckpt_file.name, "iteration": 0, "loss": None})
        
        checkpoints_info.sort(key=lambda x: x['iteration'])
        return checkpoint_paths, checkpoints_info
    
    def _process_checkpoint(
        self, ckpt_file: Path, model_meta: sly.ProjectMeta, 
        model_name: str, loss_lookup: dict
    ) -> Tuple[str, dict]:
        """Process single checkpoint file - READ ONLY, no modification"""
        state_dict = torch.load(str(ckpt_file), map_location='cpu', weights_only=False)
        
        # Extract iteration
        iteration = state_dict.get('iter', state_dict.get('epoch', 0))
        if iteration == 0:
            match = re.search(r'iter[_\s](\d+)', ckpt_file.name)
            if match:
                iteration = int(match.group(1))
        
        # Get loss from lookup or checkpoint
        loss = loss_lookup.get(iteration)
        if loss is None and 'meta' in state_dict:
            loss = state_dict.get('meta', {}).get('hook_msgs', {}).get('loss')
        
        # Copy checkpoint AS IS - no modifications
        dest_path = self.checkpoints_dir / ckpt_file.name
        shutil.copy2(str(ckpt_file), str(dest_path))
        
        return f"checkpoints/{ckpt_file.name}", {
            "name": ckpt_file.name,
            "iteration": iteration,
            "loss": loss
        }
    
    def _build_loss_lookup(self) -> dict:
        """Build iteration->loss lookup with linear interpolation"""
        loss_history = self._parse_tensorboard_logs()
        loss_lookup = {}
        
        if not loss_history:
            return loss_lookup
        
        # Find loss metric
        loss_key = 'loss' if 'loss' in loss_history else next(
            (k for k in loss_history if 'loss' in k.lower()), None
        )
        
        if not loss_key:
            return loss_lookup
        
        # Get sorted entries
        entries = sorted(loss_history[loss_key], key=lambda x: x['step'])
        
        if not entries:
            return loss_lookup
        
        # Add exact values
        for entry in entries:
            loss_lookup[entry['step']] = entry['value']
        
        # Interpolate gaps
        for i in range(len(entries) - 1):
            step1, value1 = entries[i]['step'], entries[i]['value']
            step2, value2 = entries[i+1]['step'], entries[i+1]['value']
            
            gap = step2 - step1
            if 1 < gap <= 100:  # Interpolate reasonable gaps only
                for j in range(1, gap):
                    interp_step = step1 + j
                    alpha = j / gap
                    loss_lookup[interp_step] = value1 + alpha * (value2 - value1)
        
        return loss_lookup
    
    def _copy_checkpoints_simple(self) -> Tuple[List[str], List[dict]]:
        """Fallback: simple copy without metadata"""
        checkpoint_paths = []
        checkpoints_info = []
        
        for ckpt_file in sorted(self.work_dir.glob("*.pth")):
            if ckpt_file.name != "latest.pth":
                shutil.copy2(str(ckpt_file), str(self.checkpoints_dir / ckpt_file.name))
                checkpoint_paths.append(f"checkpoints/{ckpt_file.name}")
                checkpoints_info.append({"name": ckpt_file.name, "iteration": 0, "loss": None})
        
        return checkpoint_paths, checkpoints_info
    
    # ============= TENSORBOARD LOGS =============
    
    def _parse_tensorboard_logs(self) -> dict:
        """Parse TensorBoard event files"""
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return {}
        
        logs_dir = self._find_logs_dir()
        if not logs_dir:
            return {}
        
        event_files = list(logs_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            logger.warning(f"No event files in {logs_dir}")
            return {}
        
        loss_history = {}
        for event_file in event_files:
            try:
                ea = EventAccumulator(str(event_file))
                ea.Reload()
                
                for tag in ea.Tags().get('scalars', []):
                    if tag not in loss_history:
                        loss_history[tag] = []
                    
                    for event in ea.Scalars(tag):
                        loss_history[tag].append({"step": event.step, "value": event.value})
            except Exception as e:
                logger.error(f"Failed to parse {event_file.name}: {e}", exc_info=True)
        
        if loss_history:
            logger.info(f"✅ Parsed {len(loss_history)} metrics from TensorBoard")
        
        return loss_history
    
    def _find_logs_dir(self) -> Optional[Path]:
        """Find TensorBoard logs directory"""
        candidates = [
            self.work_dir / 'vis_data',
            self.work_dir / 'tf_logs',
            self.output_dir / 'logs'
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        # Search recursively
        for candidate in self.work_dir.rglob("vis_data"):
            if candidate.is_dir() and list(candidate.glob("events.out.tfevents.*")):
                return candidate
        
        logger.warning(f"No logs found in {self.work_dir}")
        return None
    
    def _copy_logs(self) -> bool:
        """Copy TensorBoard logs to output directory"""
        logs_src = self._find_logs_dir()
        
        if logs_src and logs_src != self.output_dir / 'logs':
            logs_dest = self.output_dir / "logs"
            if logs_dest.exists():
                shutil.rmtree(logs_dest)
            shutil.copytree(logs_src, logs_dest)
            logger.info(f"✅ Logs copied from {logs_src}")
            return True
        
        return False
    
    # ============= FILE GENERATION =============
    
    def _generate_auxiliary_files(self, model_meta: sly.ProjectMeta) -> Dict[str, str]:
        """Generate all auxiliary files (config, meta, hyperparams, link)"""
        model_files = {}
        
        # Config
        if os.path.exists(self.config_file):
            config_dest = self.output_dir / os.path.basename(self.config_file)
            shutil.copy2(self.config_file, config_dest)
            model_files["config"] = config_dest.name
        
        # Model meta
        sly_json.dump_json_file(
            model_meta.to_json(), 
            str(self.output_dir / "model_meta.json")
        )
        
        # Hyperparameters
        hyperparams = LiveTrainingGenerator.parse_hyperparameters(self.config_file) \
            if os.path.exists(self.config_file) else {}
        
        
        with open(self.output_dir / "hyperparameters.yaml", 'w') as f:
            yaml.dump(hyperparams, f, default_flow_style=False)
        
        # App link
        with open(self.output_dir / "open_app.lnk", 'w') as f:
            f.write(f"/apps/sessions/{self.task_id}")
        
        logger.info("✅ Auxiliary files generated")
        return model_files
    
    def _count_dataset_images(self) -> int:
        """Count number of images in training dataset"""
        images_dir = self.work_dir.parent / 'images' / 'train'
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return 0
        
        # Support various image extensions (case-insensitive)
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        count = 0
        for ext in image_extensions:
            count += len(list(images_dir.glob(f'*{ext}')))
        
        logger.info(f"Found {count} training images in {images_dir}")
        return count
    
    def _get_training_image_ids(self) -> List[int]:
        """Get image IDs from training directory"""
        images_dir = self.work_dir.parent / 'images' / 'train'
        
        if not images_dir.exists():
            logger.warning("Training images directory not found")
            return []
        
        # Get image filenames
        image_names = []
        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_names.append(img_file.name)
        
        if not image_names:
            logger.warning("No training images found")
            return []
        
        # Get all images from project datasets and build name->id mapping
        image_ids = []
        try:
            for dataset_info in self.api.dataset.get_list(self.project_id):
                dataset_images = self.api.image.get_list(dataset_info.id)
                
                for img_info in dataset_images:
                    if img_info.name in image_names:
                        image_ids.append(img_info.id)
            
            logger.info(f"Found {len(image_ids)} image IDs from {len(image_names)} files")
            return image_ids
            
        except Exception as e:
            logger.error(f"Failed to get image IDs: {e}", exc_info=True)
            return []

    # ============= EXPERIMENT INFO =============
    
    def _prepare_experiment_info(
        self,
        model_meta: sly.ProjectMeta,
        model_config: dict,
        start_time: str,
        checkpoints: List[str],
        best_checkpoint: str,
        remote_dir: str,
        model_files: dict
    ) -> dict:
        """Prepare experiment_info dictionary"""
        project_info = self.api.project.get_info_by_id(self.project_id)
        project_preview = project_info.image_preview_url
    
        project_version = None
        if project_info:
            project_version = project_info.version

        # Count actual dataset size
        train_size = self._count_dataset_images()
        experiment_info = {
            "experiment_name": f"Live Training {self.task_type.capitalize()} - Task {self.task_id}",
            "framework_name": self.framework_name,
            "model_name": self.model_name,
            "base_checkpoint": model_config.get("backbone", "N/A"),
            "base_checkpoint_link": None,
            "task_type": self.task_type,
            "project_id": self.project_id, 
            "project_version": project_version,
            "task_id": self.task_id,
            "model_files": model_files,
            "checkpoints": checkpoints,
            "best_checkpoint": best_checkpoint,
            "export": {},
            "model_meta": "model_meta.json",
            "hyperparameters": "hyperparameters.yaml",
            "hyperparameters_id": None,
            "artifacts_dir": remote_dir,
            "datetime": start_time,
            "experiment_report_id": None,
            "evaluation_report_link": None,
            "evaluation_metrics": {},
            "primary_metric": None,
            "logs": {"type": "tensorboard", "link": f"{remote_dir}logs/"},
            "device": self._get_device_name(),
            "training_duration": self._calculate_duration(start_time),
            "train_collection_id": None,
            "val_collection_id": None,
            "project_preview": project_preview
        }

        if project_version is not None:
            experiment_info["project_version"] = project_version
        # Add train_size only if we found images
        if train_size > 0:
            experiment_info["train_size"] = train_size
            experiment_info["val_size"] = 0
            logger.info(f"📊 Dataset size: train={train_size}, val=0")
        
        return experiment_info
        
    def _calculate_duration(self, start_time: str) -> str:
        """Calculate training duration"""
        try:
            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            duration_sec = (datetime.now() - start_dt).total_seconds()
            hours, minutes = int(duration_sec // 3600), int((duration_sec % 3600) // 60)
            return f"{hours}h {minutes}m"
        except:
            return "N/A"
    
    def _get_device_name(self) -> str:
        """Get actual GPU device name or 'cpu'"""
        if not os.path.exists("/dev/nvidia0"):
            return "cpu"
        
        try:
            if torch.cuda.is_available():
                # Get device used for training (default 0)
                device_id = int(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
                device_name = torch.cuda.get_device_name(device_id)
                return device_name
        except Exception as e:
            logger.warning(f"Failed to get GPU name: {e}")
        
        return "cuda"
    # ============= MAIN UPLOAD =============
    
    def upload_artifacts(
        self,
        model_meta: sly.ProjectMeta,
        model_config: dict,
        start_time: str,
        initial_samples: int = 0,
        samples_added: int = 0
    ) -> str:
        """Upload all artifacts and generate report"""
        logger.info("📦 Starting artifacts upload...")
        
        # Prepare remote directory
        project_info = self.api.project.get_info_by_id(self.project_id)
        project_name = project_info.name if project_info else "unknown"
        
        remote_dir = f"/experiments/live_training/{self.project_id}_{project_name}/{self.task_id}_{self.framework_name}/"
        logger.info(f"📂 Remote: {remote_dir}")
        
        # Prepare checkpoints
        checkpoints, checkpoints_info = self._prepare_checkpoints(
            model_meta, self.model_name
        )

        if not checkpoints:
            raise ValueError("No checkpoints found!")
        
        best_checkpoint = next((c for c in checkpoints if 'best' in c.lower()), checkpoints[-1])
        best_checkpoint = os.path.basename(best_checkpoint)
        
        # Copy logs and parse metrics
        self._copy_logs()
        loss_history = self._parse_tensorboard_logs()
        
        # Generate auxiliary files
        model_files = self._generate_auxiliary_files(model_meta)
        
        # Upload to Team Files
        logger.info("⬆️  Uploading...")
        
        self.api.file.upload_directory_fast(
            team_id=self.team_id,
            local_dir=str(self.output_dir),
            remote_dir=remote_dir
        )
        
        # Prepare experiment_info
        experiment_info = self._prepare_experiment_info(
            model_meta, model_config, start_time,
            checkpoints, best_checkpoint, remote_dir, model_files
        )
        
        # Get hyperparameters file_id
        hyperparams_info = self.api.file.get_info_by_path(
            self.team_id, os.path.join(remote_dir, "hyperparameters.yaml")
        )
        if hyperparams_info:
            experiment_info["hyperparameters_id"] = hyperparams_info.id
        
        # Generate report
        report_url = self._generate_report(
            experiment_info, model_meta, model_config,
            remote_dir, checkpoints_info, loss_history,
            initial_samples, samples_added
        )

        # Mark that report exists (CRITICAL for experiments table)
        experiment_info["has_report"] = True
        response = self.api.task.set_output_experiment(self.task_id, experiment_info)
        return report_url
    
    def _generate_report(
        self,
        experiment_info: dict,
        model_meta: sly.ProjectMeta,
        model_config: dict,
        remote_dir: str,
        checkpoints_info: list,
        loss_history: dict,
        initial_samples: int,
        samples_added: int
    ) -> str:
        """Generate and upload Live Training report"""
        hyperparams = LiveTrainingGenerator.parse_hyperparameters(self.config_file) \
            if os.path.exists(self.config_file) else {}
        
        # Get actual dataset sizes
        train_size = experiment_info.get("train_size", 0)
        val_size = experiment_info.get("val_size", 0)
        
        session_info = {
            "session_id": self.task_id,
            "session_name": experiment_info["experiment_name"],
            "project_id": self.project_id,
            "start_time": experiment_info["datetime"],
            "duration": experiment_info["training_duration"],
            "artifacts_dir": remote_dir,
            "logs_dir": f"{remote_dir}logs/",
            "checkpoints": checkpoints_info,
            "loss_history": loss_history,
            "hyperparameters": hyperparams,
            "status": "completed",
            "device": experiment_info["device"],
            # Correct dataset info
            "dataset_size": train_size,
            "initial_samples": initial_samples,
            "samples_added": samples_added,
            "final_size": train_size,
            "train_size": train_size,
            "val_size": val_size,
        }

        report_dir = self.work_dir / "live_training_report"
        report_dir.mkdir(exist_ok=True)
        
        generator = LiveTrainingGenerator(
            api=self.api,
            session_info=session_info,
            model_config=model_config,
            model_meta=model_meta,
            output_dir=str(report_dir),
            team_id=self.team_id,
            task_type=self.task_type,
        )
        
        generator.generate()
        file_info = generator.upload_to_artifacts(os.path.join(remote_dir, "visualization"))
        
        report_id = file_info if isinstance(file_info, int) else getattr(file_info, 'id', file_info)
        experiment_info["experiment_report_id"] = report_id
        
        logger.info(f"🔗 Report URL: {self.api.server_address}/nn/experiments/{report_id}")
        
        return f"{self.api.server_address}/nn/experiments/{report_id}"