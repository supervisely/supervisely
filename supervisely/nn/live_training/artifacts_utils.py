import os
import shutil
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import supervisely as sly
from supervisely import logger
from supervisely.template.live_training.live_training_generator import LiveTrainingGenerator
import supervisely.io.json as sly_json
from supervisely.nn.live_training.helpers import ClassMap
import yaml


def upload_artifacts(
    api: sly.Api,
    session_info: dict,
    artifacts: dict,
) -> str:
    """
    Upload artifacts to Team Files and generate experiment report.

    Args:
        session_info: Training session context
            - team_id: Team ID
            - task_id: Task ID
            - project_id: Project ID
            - framework_name: Framework name
            - task_type: Task type
            - class_map: Model class map
            - start_time: Training start time string
            - train_size: Final dataset size
            - initial_samples: Number of initial samples
            

        artifacts: Framework-specific artifacts
            - checkpoint_path: Path to checkpoint file
            - checkpoint_info: Dict with {name, iteration, loss}
            - config_path: Path to config file
            - logs_dir: Path to TensorBoard logs or None
            - model_config: Model configuration dict
            - loss_history: Dict with loss history

    Returns:
        report_url: URL to experiment report
    """
    logger.info("Starting artifacts upload")

    # Unpack session_info
    team_id = session_info['team_id']
    task_id = session_info['task_id']
    project_id = session_info['project_id']
    framework_name = session_info['framework_name']
    task_type = session_info['task_type']
    class_map: ClassMap = session_info['class_map']
    model_meta = sly.ProjectMeta(obj_classes=class_map.obj_classes)
    start_time = session_info['start_time']
    train_size = session_info['train_size']
    initial_samples = session_info.get('initial_samples', 0)
    
    # Unpack artifacts
    checkpoint_path = artifacts['checkpoint_path']
    checkpoint_info = artifacts['checkpoint_info']
    config_path = artifacts['config_path']
    logs_dir = artifacts.get('logs_dir')
    model_config = artifacts['model_config']

    work_dir = Path(os.path.dirname(checkpoint_path)).parent
    output_dir = work_dir / "upload_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    project_info = api.project.get_info_by_id(project_id)
    project_name = project_info.name if project_info else "unknown"
    model_name = f"Live training - {project_name}"
    model_config['model_name'] = model_name
    remote_dir = f"/experiments/live_training/{project_id}_{project_name}/{task_id}_{framework_name}/"
    logger.info(f"Remote directory: {remote_dir}")

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoint_dest = checkpoints_dir / checkpoint_info["name"]
    shutil.copy2(checkpoint_path, checkpoint_dest)

    state_json_src = checkpoint_path.replace('.pth', '_state.json')
    if os.path.exists(state_json_src):
        state_json_dest = str(checkpoint_dest).replace('.pth', '_state.json')
        shutil.copy2(state_json_src, state_json_dest)

    if config_path and os.path.exists(config_path):
        config_dest = output_dir / os.path.basename(config_path)
        shutil.copy2(config_path, config_dest)
        model_files = {"config": config_dest.name}
    else:
        model_files = {}

    sly_json.dump_json_file(
        model_meta.to_json(),
        str(output_dir / "model_meta.json")
    )

    hyperparams = {}
    if config_path and os.path.exists(config_path):
        try:
            hyperparams = LiveTrainingGenerator.parse_hyperparameters(config_path)
        except Exception as e:
            logger.warning(f"Failed to parse hyperparameters: {e}")

    with open(output_dir / "hyperparameters.yaml", 'w') as f:
        yaml.dump(hyperparams, f, default_flow_style=False)

    with open(output_dir / "open_app.lnk", 'w') as f:
        f.write(f"/apps/sessions/{task_id}")

    if logs_dir and os.path.exists(logs_dir):
        logs_dest = output_dir / "logs"
        if logs_dest.exists():
            shutil.rmtree(logs_dest)
        shutil.copytree(logs_dir, logs_dest)
        logger.info(f"Logs copied from {logs_dir}")
        has_logs = True
    else:
        logger.warning("No logs provided")
        has_logs = False

    logger.info("Uploading to Team Files")
    api.file.upload_directory_fast(
        team_id=team_id,
        local_dir=str(output_dir),
        remote_dir=remote_dir
    )

    experiment_info = {
        "experiment_name": f"Live Training {task_type.capitalize()} - Task {task_id}",
        "framework_name": framework_name,
        "model_name": model_name,
        "base_checkpoint": None,
        "base_checkpoint_link": None,
        "task_type": task_type,
        "project_id": project_id,
        "project_version": project_info.version if project_info else None,
        "task_id": task_id,
        "model_files": model_files,
        "checkpoints": [f"checkpoints/{checkpoint_info['name']}"],
        "best_checkpoint": checkpoint_info['name'],
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
        "logs": {"type": "tensorboard", "link": f"{remote_dir}logs/"} if has_logs else None,
        "device": get_device_name(),
        "training_duration": calculate_duration(start_time),
        "train_collection_id": None,
        "val_collection_id": None,
        "project_preview": project_info.image_preview_url if project_info else None,
        "train_size": train_size,
        "initial_samples": initial_samples,
        "val_size": 0,
    }

    checkpoints_info = [checkpoint_info]
    loss_history = artifacts.get('loss_history', {})

    session_info = {
        "session_id": task_id,
        "session_name": experiment_info["experiment_name"],
        "project_id": project_id,
        "start_time": experiment_info["datetime"],
        "duration": experiment_info["training_duration"],
        "artifacts_dir": remote_dir,
        "logs_dir": f"{remote_dir}logs/" if has_logs else None,
        "checkpoints": checkpoints_info,
        "loss_history": loss_history,
        "hyperparameters": hyperparams,
        "status": "completed",
        "device": experiment_info["device"],
        "dataset_size": train_size,
        "initial_samples": 0,
        "samples_added": 0,
        "final_size": train_size,
        "train_size": train_size,
        "initial_samples": initial_samples,
        "val_size": 0,
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

    logger.info(f"Report URL: {report_url}")

    experiment_info["has_report"] = True
    experiment_info["experiment_report_id"] = int(report_url.split('/')[-1])
    response = api.task.set_output_experiment(task_id, experiment_info)

    return report_url


def calculate_duration(start_time: str) -> str:
    """Calculate training duration in 'Xh Ym' format."""
    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        duration_sec = (datetime.now() - start_dt).total_seconds()
        hours = int(duration_sec // 3600)
        minutes = int((duration_sec % 3600) // 60)
        return f"{hours}h {minutes}m"
    except:
        return "N/A"


def get_device_name() -> str:
    """Get GPU device name or 'cpu'."""
    import torch # pylint: disable=import-error
    if not os.path.exists("/dev/nvidia0"):
        return "cpu"

    try:
        if torch.cuda.is_available():
            device_id = int(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
            return torch.cuda.get_device_name(device_id)
    except Exception as e:
        logger.warning(f"Failed to get GPU name: {e}")

    return "cuda"