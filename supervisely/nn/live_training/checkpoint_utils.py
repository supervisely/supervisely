import os
import os.path as osp
from pathlib import Path
from typing import Optional, List, Dict
import torch
import supervisely as sly
from supervisely import logger


def resolve_checkpoint(
    api: sly.Api,
    team_id: int,
    work_dir: str,
    num_classes: int,
    backbone_type: str,
    project_id: int,
    framework_name: str,
    checkpoint_mode: str = "scratch",
    resume_dataset: bool = None,
    selected_experiment_task_id: Optional[int] = None,
    class_names: List[str] = None,
) -> tuple[Optional[str], bool, Optional[Dict], Optional[Dict]]:
    """Resolve checkpoint based on checkpoint_mode parameter.
    
    Framework-agnostic checkpoint resolution. Returns dataset_metadata which
    should be processed by app-specific restore_dataset_from_checkpoint().
    
    Modes:
        - "scratch": Start from pretrained weights (no checkpoint loading)
        - "finetune": Start from user checkpoint
        - "continue": Continue from previous experiment (loads checkpoint + optionally dataset)
    
    Parameters:
        checkpoint_mode: Mode selection ("scratch", "finetune", or "continue")
        resume_dataset: Restore dataset when mode="continue" (default: None)
        selected_experiment_task_id: Task ID to continue from (required for "continue" mode)
        class_names: List of class names (without background)
    
    Returns:
        tuple: (checkpoint_path, should_resume, dataset_metadata, class_mapping)
        - checkpoint_path: Local path to checkpoint file (or None)
        - should_resume: Whether to resume training
        - dataset_metadata: Dict with 'project_id', 'image_ids', 'classes' (or None)
        - class_mapping: Dict for class reordering (or None)
    """
    
    logger.info(f"Checkpoint mode: {checkpoint_mode}")
    
    if checkpoint_mode == "scratch":
        logger.info("Starting from pretrained weights (scratch mode)")
        return None, False, None, None
    
    elif checkpoint_mode == "finetune":
        if selected_experiment_task_id is None:
            raise ValueError(
                "selected_experiment_task_id must be provided when checkpoint_mode='finetune'"
            )
        logger.info(f"Finetune mode: loading checkpoint from task_id={selected_experiment_task_id}")
        
        checkpoint_path = _find_checkpoint_by_task_id(
            api, team_id, selected_experiment_task_id, framework_name
        )
        
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found for task_id={selected_experiment_task_id}")
        
        local_checkpoint = download_checkpoint_from_team_files(api, team_id, checkpoint_path, work_dir)
        checkpoint = torch.load(local_checkpoint, map_location='cpu', weights_only=False)

        # Get classes from dataset_metadata (without background)
        dataset_metadata = checkpoint.get('dataset_metadata', {})
        saved_classes = dataset_metadata.get('classes', [])

        if not saved_classes:
            raise ValueError(
                "Checkpoint does not contain class information in dataset_metadata. "
                "This checkpoint was created with an older version."
            )

        logger.info(f"Saved classes in checkpoint (without bg): {saved_classes}, current classes: {class_names}")
        
        saved_classes_set = set(saved_classes)
        current_classes_set = set(class_names)
        
        if saved_classes_set == current_classes_set:
            if saved_classes != class_names:
                logger.info("The class order is different. Reordering classes")
                class_mapping = {cls: idx for idx, cls in enumerate(saved_classes)}
                logger.info(f"Class mapping: {class_mapping}")
                return local_checkpoint, False, None, class_mapping
            else:
                logger.info("Class names match exactly")
                return local_checkpoint, False, None, None
            
        elif len(saved_classes) == len(class_names) and saved_classes_set != current_classes_set:
            # Remove classification head weights from checkpoint
            logger.info("Class names differ but number of classes match. Removing classification head weights")
            state_dict = checkpoint.get('state_dict', {})
            keys_to_remove = []
            
            for key in state_dict.keys():
                # Remove decode_head and auxiliary_head classification layers
                if 'decode_head' in key or 'auxiliary_head' in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del state_dict[key]
            
            logger.info(f"Removed {len(keys_to_remove)} classification head parameters")
            
            # Save modified checkpoint
            modified_checkpoint_path = local_checkpoint.replace('.pth', '_headless.pth')
            torch.save(checkpoint, modified_checkpoint_path)
            return modified_checkpoint_path, False, None, None

        else:
            logger.info("Class names and number of classes do not match. Removing classification head weights")
            return local_checkpoint, False, None, None
    
    elif checkpoint_mode == "continue":
        resume_dataset = True if resume_dataset is None else resume_dataset  # Default to True
        if selected_experiment_task_id is None:
            raise ValueError(
                "selected_experiment_task_id must be provided when checkpoint_mode='continue'"
            )
        
        logger.info(f"Continue mode: loading checkpoint from task_id={selected_experiment_task_id}")
        logger.info(f"Resume dataset: {resume_dataset}")
        
        checkpoint_path = _find_checkpoint_by_task_id(
            api, team_id, selected_experiment_task_id, framework_name
        )
        
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found for task_id={selected_experiment_task_id}")
        
        local_checkpoint = download_checkpoint_from_team_files(api, team_id, checkpoint_path, work_dir)
        
        is_compatible, reason, dataset_meta = _check_compatibility(
            local_checkpoint, num_classes, backbone_type, project_id,
            expected_class_names=class_names,
            allow_cross_project=True
        )
        
        if not is_compatible:
            raise ValueError(f"Selected experiment is incompatible: {reason}")
        
        logger.info(f"Checkpoint loaded from task_id={selected_experiment_task_id}")
        
        if dataset_meta and resume_dataset:
            saved_project_id = dataset_meta.get('project_id')
            dataset_size = dataset_meta.get('dataset_size', 0)
            logger.info(f"Dataset metadata ready for restoration: {dataset_size} images from project {saved_project_id}")
            # NOTE: dataset_meta should be passed to app-specific restore_dataset_from_checkpoint()
            return local_checkpoint, True, dataset_meta, None
        else:
            if not resume_dataset:
                logger.info("Dataset restoration disabled (resume_dataset=False)")
            return local_checkpoint, True, None, None
    
    else:
        raise ValueError(
            f"Invalid checkpoint_mode='{checkpoint_mode}'. "
            f"Valid values: 'scratch', 'finetune', 'continue'"
        )


def download_checkpoint_from_team_files(
    api: sly.Api,
    team_id: int,
    remote_checkpoint_path: str,
    local_work_dir: str
) -> str:
    """Download checkpoint from Team Files to local directory."""
    local_checkpoints_dir = osp.join(local_work_dir, 'downloaded_checkpoints')
    os.makedirs(local_checkpoints_dir, exist_ok=True)
    
    checkpoint_name = osp.basename(remote_checkpoint_path)
    local_checkpoint_path = osp.join(local_checkpoints_dir, checkpoint_name)
    
    logger.info(f"Downloading checkpoint...")
    logger.info(f"Remote: {remote_checkpoint_path}")
    logger.info(f"Local: {local_checkpoint_path}")
    
    remote_file_info = api.file.get_info_by_path(team_id, remote_checkpoint_path)
    expected_size = remote_file_info.sizeb
    
    api.file.download(
        team_id=team_id,
        remote_path=remote_checkpoint_path,
        local_save_path=local_checkpoint_path
    )
    
    actual_size = os.path.getsize(local_checkpoint_path)
    if actual_size != expected_size:
        if os.path.exists(local_checkpoint_path):
            os.remove(local_checkpoint_path)
        raise ValueError(
            f"Downloaded file size mismatch! "
            f"Expected: {expected_size} bytes, Got: {actual_size} bytes"
        )
    
    logger.info(f"Checkpoint downloaded ({actual_size / 1024 / 1024:.1f} MB)")
    return local_checkpoint_path


def _find_latest_checkpoint(
    api: sly.Api,
    team_id: int,
    project_id: int,
    framework_name: str = "Mask2Former"
) -> Optional[tuple[str, int]]:
    """Find the latest checkpoint for the current project."""
    project_info = api.project.get_info_by_id(project_id)
    if not project_info:
        logger.warning(f"Project {project_id} not found")
        return None
    
    experiments_base = f"/experiments/live_training/{project_id}_{project_info.name}/"
    
    logger.info(f"Searching for experiments in: {experiments_base}")
    
    if not api.file.dir_exists(team_id, experiments_base):
        logger.info("No experiments directory found")
        return None
    
    items = api.file.list(team_id, experiments_base, recursive=False, return_type='fileinfo')
    
    experiment_dirs = []
    for item in items:
        if not item.is_dir:
            continue
        
        if f"_{framework_name}" not in item.name:
            continue
        
        task_id_str = item.name.split('_')[0]
        if not task_id_str.isdigit():
            continue
        
        experiment_dirs.append({
            'path': item.path,
            'task_id': int(task_id_str),
        })
    
    if not experiment_dirs:
        logger.info("No previous experiments found")
        return None
    
    experiment_dirs.sort(key=lambda x: x['task_id'], reverse=True)
    latest_experiment = experiment_dirs[0]
    
    logger.info(f"Found {len(experiment_dirs)} experiment(s), latest: task_id={latest_experiment['task_id']}")
    
    checkpoints_dir = latest_experiment['path'] + 'checkpoints/'
    checkpoint_files = _find_checkpoint_files(api, team_id, checkpoints_dir)
    
    if not checkpoint_files:
        logger.warning("No checkpoint files found")
        return None
    
    last_checkpoint = checkpoint_files[0]
    iteration = _extract_iteration(last_checkpoint.name)
    
    logger.info(f"Found checkpoint: {last_checkpoint.path} (iteration {iteration})")
    
    return last_checkpoint.path, latest_experiment['task_id']


def _find_checkpoint_by_task_id(
    api: sly.Api,
    team_id: int,
    task_id: int,
    framework_name: str = "Mask2Former"
) -> Optional[str]:
    """Find checkpoint path by task_id for cross-project loading."""
    experiments_base = "/experiments/live_training/"
    
    project_dirs = api.file.list(team_id, experiments_base, recursive=False, return_type='fileinfo')
    
    for project_dir in project_dirs:
        if not project_dir.is_dir:
            continue
        
        task_dirs = api.file.list(team_id, project_dir.path, recursive=False, return_type='fileinfo')
        
        for task_dir in task_dirs:
            if not task_dir.is_dir:
                continue
            
            if task_dir.name.startswith(f"{task_id}_") and f"_{framework_name}" in task_dir.name:
                checkpoints_dir = task_dir.path + 'checkpoints/'
                checkpoint_files = _find_checkpoint_files(api, team_id, checkpoints_dir)
                
                if checkpoint_files:
                    return checkpoint_files[0].path
    
    return None


def _find_checkpoint_files(api: sly.Api, team_id: int, checkpoints_dir: str) -> List:
    """Find all checkpoint files in the given directory, sorted by iteration."""
    if not api.file.dir_exists(team_id, checkpoints_dir):
        return []
    
    files = api.file.list(team_id, checkpoints_dir, recursive=False, return_type='fileinfo')
    
    checkpoint_files = []
    for file_info in files:
        if file_info.name.endswith('.pth') and file_info.name != 'latest.pth':
            iteration = _extract_iteration(file_info.name)
            checkpoint_files.append((file_info, iteration))
    
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in checkpoint_files]


def _extract_iteration(filename: str) -> int:
    """Extract iteration number from checkpoint filename."""
    import re
    match = re.search(r'iter[_\s](\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def _check_compatibility(
    checkpoint_path: str,
    expected_num_classes: int,
    expected_backbone: str,
    expected_project_id: int,
    expected_class_names: Optional[List[str]] = None,
    allow_cross_project: bool = False
) -> tuple[bool, str, Optional[Dict]]:
    """Check checkpoint compatibility.
    
    Validation:
        - dataset_metadata must exist (for resume)
        - class names must match (if expected_class_names provided)
        - project_id must match (if allow_cross_project=False)
    
    Returns:
        tuple: (is_compatible, reason, dataset_metadata)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check for dataset_metadata (required for dataset restoration)
    dataset_metadata = checkpoint.get('dataset_metadata', None)
    
    if not dataset_metadata:
        return False, (
            "Checkpoint does not contain dataset metadata (image_ids). "
            "This checkpoint was created with an older version"
        ), None
    
    saved_project_id = dataset_metadata.get('project_id')
    saved_class_names = dataset_metadata.get('classes', [])
    
    if allow_cross_project:
        if expected_class_names is not None and saved_class_names:
            if set(saved_class_names) != set(expected_class_names):
                return False, f"class names mismatch (saved: {saved_class_names}, expected: {expected_class_names})", None
        
        if saved_project_id != expected_project_id:
            logger.warning(
                f"Cross-project resume: loading checkpoint from project {saved_project_id} "
                f"into project {expected_project_id}"
            )
    else:
        if saved_project_id != expected_project_id:
            return False, f"project_id mismatch (saved: {saved_project_id}, expected: {expected_project_id})", None
    
    return True, "compatible", dataset_metadata