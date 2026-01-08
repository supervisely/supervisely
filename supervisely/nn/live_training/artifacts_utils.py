import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import supervisely as sly
from supervisely import logger
from supervisely.template.live_training.live_training_generator import LiveTrainingGenerator
import supervisely.io.json as sly_json
import yaml
import re



# ============= CHECKPOINTS =============

def prepare_checkpoints(
    work_dir: Path,
    output_dir: Path,
    model_meta: sly.ProjectMeta,
    model_name: str
) -> Tuple[List[str], List[dict]]:
    """
    Prepare checkpoints with metadata (iteration, loss).
    
    Returns:
        checkpoint_paths: List of relative paths like "checkpoints/iter_100.pth"
        checkpoints_info: List of dicts with {name, iteration, loss}
    """
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    loss_lookup = build_loss_lookup(work_dir)
    
    checkpoint_paths = []
    checkpoints_info = []
    
    for ckpt_file in sorted(work_dir.glob("*.pth")):
        if ckpt_file.name == "latest.pth":
            continue
        
        try:
            path, info = process_checkpoint(
                ckpt_file, checkpoints_dir, loss_lookup
            )
            checkpoint_paths.append(path)
            checkpoints_info.append(info)
            logger.info(f"✅ {ckpt_file.name}: iter={info['iteration']}, loss={info['loss']}")
        except Exception as e:
            logger.error(f"Failed to process {ckpt_file.name}: {e}")
            raise
    
    if not checkpoint_paths:
        raise ValueError("No checkpoints found in work_dir")
    
    checkpoints_info.sort(key=lambda x: x['iteration'])
    return checkpoint_paths, checkpoints_info


def process_checkpoint(
    ckpt_file: Path,
    checkpoints_dir: Path,
    loss_lookup: dict
) -> Tuple[str, dict]:
    """Process single checkpoint - extract metadata and copy to output"""
    import torch  # pylint: disable=import-error
    state_dict = torch.load(str(ckpt_file), map_location='cpu', weights_only=False)
    
    # Extract iteration
    iteration = state_dict.get('iter', state_dict.get('epoch', 0))
    if iteration == 0:
        match = re.search(r'iter[_\s](\d+)', ckpt_file.name)
        if match:
            iteration = int(match.group(1))
    
    # Get loss
    loss = loss_lookup.get(iteration)
    if loss is None and 'meta' in state_dict:
        loss = state_dict.get('meta', {}).get('hook_msgs', {}).get('loss')
    
    # Copy checkpoint
    dest_path = checkpoints_dir / ckpt_file.name
    shutil.copy2(str(ckpt_file), str(dest_path))
    
    return f"checkpoints/{ckpt_file.name}", {
        "name": ckpt_file.name,
        "iteration": iteration,
        "loss": loss
    }


def build_loss_lookup(work_dir: Path) -> dict:
    """
    Build iteration->loss mapping with linear interpolation.
    Returns empty dict if no logs found.
    """
    loss_history = parse_tensorboard_logs(work_dir)
    loss_lookup = {}
    
    if not loss_history:
        return loss_lookup
    
    # Find loss metric
    loss_key = 'loss' if 'loss' in loss_history else next(
        (k for k in loss_history if 'loss' in k.lower()), None
    )
    
    if not loss_key:
        return loss_lookup
    
    entries = sorted(loss_history[loss_key], key=lambda x: x['step'])
    if not entries:
        return loss_lookup
    
    # Exact values
    for entry in entries:
        loss_lookup[entry['step']] = entry['value']
    
    # Interpolate gaps
    for i in range(len(entries) - 1):
        step1, value1 = entries[i]['step'], entries[i]['value']
        step2, value2 = entries[i+1]['step'], entries[i+1]['value']
        
        gap = step2 - step1
        if 1 < gap <= 100:
            for j in range(1, gap):
                interp_step = step1 + j
                alpha = j / gap
                loss_lookup[interp_step] = value1 + alpha * (value2 - value1)
    
    return loss_lookup


# ============= TENSORBOARD LOGS =============

def parse_tensorboard_logs(work_dir: Path) -> dict:
    """Parse TensorBoard event files and return all scalar metrics"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("TensorBoard not installed, skipping logs parsing")
        return {}
    
    logs_dir = find_logs_dir(work_dir)
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
            logger.error(f"Failed to parse {event_file.name}: {e}")
    
    if loss_history:
        logger.info(f"✅ Parsed {len(loss_history)} metrics from TensorBoard")
    
    return loss_history


def find_logs_dir(work_dir: Path) -> Optional[Path]:
    """Find TensorBoard logs directory"""
    candidates = [
        work_dir / 'vis_data',
        work_dir / 'tf_logs',
    ]
    
    for path in candidates:
        if path.exists() and list(path.glob("events.out.tfevents.*")):
            return path
    
    # Search recursively
    for candidate in work_dir.rglob("vis_data"):
        if candidate.is_dir() and list(candidate.glob("events.out.tfevents.*")):
            return candidate
    
    logger.warning(f"No TensorBoard logs found in {work_dir}")
    return None


def copy_logs(work_dir: Path, output_dir: Path) -> bool:
    """Copy TensorBoard logs to output directory"""
    logs_src = find_logs_dir(work_dir)
    
    if not logs_src:
        return False
    
    logs_dest = output_dir / "logs"
    if logs_dest.exists():
        shutil.rmtree(logs_dest)
    
    shutil.copytree(logs_src, logs_dest)
    logger.info(f"✅ Logs copied from {logs_src}")
    return True


# ============= FILE GENERATION =============

def generate_auxiliary_files(
    output_dir: Path,
    config_file: str,
    model_meta: sly.ProjectMeta,
    task_id: int
) -> Dict[str, str]:
    """
    Generate config, model_meta.json, hyperparameters.yaml, open_app.lnk
    
    Returns:
        model_files: Dict with generated file names
    """
    model_files = {}
    
    # Config
    if os.path.exists(config_file):
        config_dest = output_dir / os.path.basename(config_file)
        shutil.copy2(config_file, config_dest)
        model_files["config"] = config_dest.name
    
    # Model meta
    sly_json.dump_json_file(
        model_meta.to_json(),
        str(output_dir / "model_meta.json")
    )
    
    # Hyperparameters
    hyperparams = LiveTrainingGenerator.parse_hyperparameters(config_file) \
        if os.path.exists(config_file) else {}
    
    with open(output_dir / "hyperparameters.yaml", 'w') as f:
        yaml.dump(hyperparams, f, default_flow_style=False)
    
    # App link
    with open(output_dir / "open_app.lnk", 'w') as f:
        f.write(f"/apps/sessions/{task_id}")
    
    logger.info("✅ Auxiliary files generated")
    return model_files


def count_dataset_images(work_dir: Path) -> int:
    """Count images in training dataset directory"""
    images_dir = work_dir.parent / 'images' / 'train'
    
    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return 0
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    count = sum(len(list(images_dir.glob(f'*{ext}'))) for ext in image_extensions)
    
    logger.info(f"Found {count} training images in {images_dir}")
    return count


# ============= EXPERIMENT INFO =============

def prepare_experiment_info(
    api: sly.Api,
    project_id: int,
    task_id: int,
    work_dir: Path,
    config_file: str,
    framework_name: str,
    model_name: str,
    task_type: str,
    model_config: dict,
    start_time: str,
    checkpoints: List[str],
    best_checkpoint: str,
    remote_dir: str,
    model_files: dict
) -> dict:
    """Build experiment_info dictionary for Supervisely experiments page"""
    project_info = api.project.get_info_by_id(project_id)
    train_size = count_dataset_images(work_dir)
    
    experiment_info = {
        "experiment_name": f"Live Training {task_type.capitalize()} - Task {task_id}",
        "framework_name": framework_name,
        "model_name": model_name,
        "base_checkpoint": model_config.get("backbone", "N/A"),
        "base_checkpoint_link": None,
        "task_type": task_type,
        "project_id": project_id,
        "project_version": project_info.version if project_info else None,
        "task_id": task_id,
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
        "device": get_device_name(),
        "training_duration": calculate_duration(start_time),
        "train_collection_id": None,
        "val_collection_id": None,
        "project_preview": project_info.image_preview_url if project_info else None
    }
    
    if train_size > 0:
        experiment_info["train_size"] = train_size
        experiment_info["val_size"] = 0
        logger.info(f"📊 Dataset size: train={train_size}, val=0")
    
    return experiment_info


def calculate_duration(start_time: str) -> str:
    """Calculate training duration in 'Xh Ym' format"""
    import torch  # pylint: disable=import-error
    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        duration_sec = (datetime.now() - start_dt).total_seconds()
        hours = int(duration_sec // 3600)
        minutes = int((duration_sec % 3600) // 60)
        return f"{hours}h {minutes}m"
    except:
        return "N/A"


def get_device_name() -> str:
    """Get GPU device name or 'cpu'"""
    import torch  # pylint: disable=import-error
    if not os.path.exists("/dev/nvidia0"):
        return "cpu"
    
    try:
        if torch.cuda.is_available():
            device_id = int(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
            return torch.cuda.get_device_name(device_id)
    except Exception as e:
        logger.warning(f"Failed to get GPU name: {e}")
    
    return "cuda"


# ============= REPORT GENERATION =============

def generate_and_upload_report(
    api: sly.Api,
    team_id: int,
    task_id: int,
    project_id: int,
    work_dir: Path,
    config_file: str,
    task_type: str,
    experiment_info: dict,
    model_meta: sly.ProjectMeta,
    model_config: dict,
    remote_dir: str,
    checkpoints_info: list,
    loss_history: dict,
    initial_samples: int,
    samples_added: int
) -> str:
    """Generate Live Training report and upload to Team Files"""
    hyperparams = LiveTrainingGenerator.parse_hyperparameters(config_file) \
        if os.path.exists(config_file) else {}
    
    train_size = experiment_info.get("train_size", 0)
    val_size = experiment_info.get("val_size", 0)
    
    session_info = {
        "session_id": task_id,
        "session_name": experiment_info["experiment_name"],
        "project_id": project_id,
        "start_time": experiment_info["datetime"],
        "duration": experiment_info["training_duration"],
        "artifacts_dir": remote_dir,
        "logs_dir": f"{remote_dir}logs/",
        "checkpoints": checkpoints_info,
        "loss_history": loss_history,
        "hyperparameters": hyperparams,
        "status": "completed",
        "device": experiment_info["device"],
        "dataset_size": train_size,
        "initial_samples": initial_samples,
        "samples_added": samples_added,
        "final_size": train_size,
        "train_size": train_size,
        "val_size": val_size,
    }
    
    report_dir = work_dir / "live_training_report"
    report_dir.mkdir(exist_ok=True)
    
    generator = LiveTrainingGenerator(
        api=api,
        session_info=session_info,
        model_config=model_config,
        model_meta=model_meta,
        output_dir=str(report_dir),
        team_id=team_id,
        task_type=task_type,
    )
    
    generator.generate()
    file_info = generator.upload_to_artifacts(os.path.join(remote_dir, "visualization"))
    
    report_id = file_info if isinstance(file_info, int) else getattr(file_info, 'id', file_info)
    report_url = f"{api.server_address}/nn/experiments/{report_id}"
    
    logger.info(f"🔗 Report URL: {report_url}")
    return report_url


# ============= MAIN ORCHESTRATOR =============

def upload_artifacts(
    api: sly.Api,
    team_id: int,
    task_id: int,
    project_id: int,
    work_dir: str,
    config_file: str,
    framework_name: str,
    model_name: str,
    task_type: str,
    model_meta: sly.ProjectMeta,
    model_config: dict,
    start_time: str,
    initial_samples: int = 0,
    samples_added: int = 0
) -> str:
    """
    Main function: prepare artifacts, upload to Team Files, generate report.
    
    Returns:
        report_url: URL to experiment report
    """
    logger.info("📦 Starting artifacts upload...")
    
    work_dir = Path(work_dir)
    output_dir = work_dir / "upload_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remote directory
    project_info = api.project.get_info_by_id(project_id)
    project_name = project_info.name if project_info else "unknown"
    remote_dir = f"/experiments/live_training/{project_id}_{project_name}/{task_id}_{framework_name}/"
    logger.info(f"📂 Remote: {remote_dir}")
    
    # Prepare checkpoints
    checkpoints, checkpoints_info = prepare_checkpoints(
        work_dir, output_dir, model_meta, model_name
    )
    best_checkpoint = next((c for c in checkpoints if 'best' in c.lower()), checkpoints[-1])
    best_checkpoint = os.path.basename(best_checkpoint)
    
    # Copy logs and parse metrics
    copy_logs(work_dir, output_dir)
    loss_history = parse_tensorboard_logs(work_dir)
    
    # Generate auxiliary files
    model_files = generate_auxiliary_files(output_dir, config_file, model_meta, task_id)
    
    # Upload to Team Files
    logger.info("⬆️  Uploading...")
    api.file.upload_directory_fast(
        team_id=team_id,
        local_dir=str(output_dir),
        remote_dir=remote_dir
    )
    
    # Prepare experiment_info
    experiment_info = prepare_experiment_info(
        api, project_id, task_id, work_dir, config_file,
        framework_name, model_name, task_type, model_config,
        start_time, checkpoints, best_checkpoint, remote_dir, model_files
    )
    
    # Get hyperparameters file_id
    hyperparams_info = api.file.get_info_by_path(
        team_id, os.path.join(remote_dir, "hyperparameters.yaml")
    )
    if hyperparams_info:
        experiment_info["hyperparameters_id"] = hyperparams_info.id
    
    # Generate report
    report_url = generate_and_upload_report(
        api, team_id, task_id, project_id, work_dir, config_file, task_type,
        experiment_info, model_meta, model_config, remote_dir,
        checkpoints_info, loss_history, initial_samples, samples_added
    )
    
    # Mark report exists and save experiment_info
    experiment_info["has_report"] = True
    experiment_info["experiment_report_id"] = int(report_url.split('/')[-1])
    api.task.set_output_experiment(task_id, experiment_info)
    
    return report_url