import os
import re
from typing import Tuple, List, Optional
import json
from supervisely import logger
from supervisely.nn.live_training.helpers import ClassMap


def find_latest_checkpoint_path(
    api,
    team_id: int,
    selected_experiment_task_id: int, 
    framework_name: str
) -> str:
    """
    Find the latest checkpoint for given experiment task.
    
    Args:
        selected_experiment_task_id: ID of experiment to load checkpoint from
    """
    experiments_base = "/experiments/live_training/"
    
    project_dirs = api.file.list(team_id, experiments_base, recursive=False, return_type='fileinfo')
    
    for project_dir in project_dirs:
        if not project_dir.is_dir:
            continue
        
        task_dirs = api.file.list(team_id, project_dir.path, recursive=False, return_type='fileinfo')
        
        for task_dir in task_dirs:
            if not task_dir.is_dir:
                continue
            
            if task_dir.name.startswith(f"{selected_experiment_task_id}_") and f"_{framework_name}" in task_dir.name:
                checkpoints_dir = task_dir.path + 'checkpoints/'
                if not api.file.dir_exists(team_id, checkpoints_dir):
                    break
                
                files = api.file.list(team_id, checkpoints_dir, recursive=False, return_type='fileinfo')
                
                checkpoint_files = []
                for file_info in files:
                    if file_info.name.endswith('.pth') and file_info.name != 'latest.pth':
                        match = re.search(r'iter[_\s](\d+)', file_info.name)
                        iteration = int(match.group(1)) if match else 0
                        checkpoint_files.append((file_info, iteration))
                
                checkpoint_files.sort(key=lambda x: x[1], reverse=True)
                checkpoint_files = [f[0] for f in checkpoint_files]
                
                if checkpoint_files:
                    return checkpoint_files[0].path
    
    raise ValueError(f"No checkpoint found for experiment task_id={selected_experiment_task_id}")

def download_checkpoint_file(
    api,
    team_id: int,
    remote_path: str,
    local_dir: str
) -> str:
    """
    Download checkpoint file from Team Files with size validation.
    
    Returns:
        local_path: Path to downloaded checkpoint file
    """
    os.makedirs(local_dir, exist_ok=True)
    
    checkpoint_name = os.path.basename(remote_path)
    local_path = os.path.join(local_dir, checkpoint_name)
    
    logger.info(f"Downloading checkpoint...")
    logger.info(f"Remote: {remote_path}")
    logger.info(f"Local: {local_path}")
    
    remote_file_info = api.file.get_info_by_path(team_id, remote_path)
    expected_size = remote_file_info.sizeb
    
    api.file.download(
        team_id=team_id,
        remote_path=remote_path,
        local_save_path=local_path
    )
    
    actual_size = os.path.getsize(local_path)
    if actual_size != expected_size:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise ValueError(
            f"Downloaded file size mismatch! "
            f"Expected: {expected_size} bytes, Got: {actual_size} bytes"
        )
    
    logger.info(f"Checkpoint downloaded ({actual_size / 1024 / 1024:.1f} MB)")
    return local_path


def load_checkpoint_metadata(checkpoint_path: str) -> Tuple[dict, List[str]]:
    """
    Load checkpoint and extract dataset metadata.
    
    Returns:
        checkpoint: Loaded torch checkpoint dict
        saved_classes: List of class names from metadata
    """
    import torch  # pylint: disable=import-error
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    dataset_metadata = checkpoint.get('dataset_metadata', {})
    if not dataset_metadata:
        raise ValueError(
            "Checkpoint does not contain dataset metadata. "
            "This checkpoint was created with an older version."
        )
    
    saved_classes = dataset_metadata.get('classes', [])
    return checkpoint, saved_classes


def validate_classes_exact_match(saved_classes: List[str], current_classes: List[str]):
    """
    Validate that saved and current classes match exactly.
    Raises ValueError if they don't match.
    """
    if set(saved_classes) != set(current_classes):
        raise ValueError(
            f"Class names in checkpoint do not match current class names. "
            f"Saved: {saved_classes}, Current: {current_classes}"
        )


def reorder_class_map(saved_classes: List[str], project_meta) -> ClassMap:
    """
    Create ClassMap with reordered classes matching checkpoint order.
    
    Returns:
        class_map: New ClassMap with correct order
    """
    class_mapping = {cls: idx for idx, cls in enumerate(saved_classes)}
    logger.info(f"Class mapping: {class_mapping}")
    return ClassMap.from_names(list(class_mapping.keys()), project_meta)


def remove_classification_head(checkpoint_path: str) -> str:
    """
    Remove classification head weights from checkpoint and save modified version.
    
    Returns:
        modified_path: Path to checkpoint without classification head
    """
    import torch  # pylint: disable=import-error
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', {})
    
    keys_to_remove = []
    for key in state_dict.keys():
        if 'decode_head' in key or 'auxiliary_head' in key:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del state_dict[key]
    
    logger.info(f"Removed {len(keys_to_remove)} classification head parameters")
    
    modified_path = checkpoint_path.replace('.pth', '_headless.pth')
    torch.save(checkpoint, modified_path)
    
    return modified_path


def resolve_checkpoint(
    checkpoint_mode: str,
    selected_experiment_task_id: Optional[int],
    class_map: ClassMap,
    project_meta,
    api,
    team_id: int,
    framework_name: str,
    work_dir: str
) -> Tuple[Optional[str], ClassMap]:
    """
    Main orchestrator function to resolve checkpoint loading based on mode.
    
    Args:
        checkpoint_mode: One of 'scratch', 'finetune', 'continue'
        selected_experiment_task_id: Task ID to load checkpoint from (required for finetune/continue)
        class_map: Current ClassMap
        project_meta: Project metadata
        api: Supervisely API instance
        team_id: Team ID
        framework_name: Framework name (e.g., 'mmseg', 'mmdet')
        work_dir: Working directory for downloaded files
    
    Returns:
        checkpoint_path: Path to checkpoint file (None for scratch mode)
        class_map: Updated ClassMap (may be reordered for finetune/continue)
    """
    current_classes = [cls.name for cls in class_map.obj_classes]
    logger.info(f"Checkpoint mode: {checkpoint_mode}")
    
    # Scratch mode - no checkpoint needed
    if checkpoint_mode == "scratch":
        logger.info("Starting from pretrained weights (scratch mode)")
        return None, class_map, None
    
    # Finetune and Continue modes require task_id
    if selected_experiment_task_id is None:
        raise ValueError(
            f"selected_experiment_task_id must be provided when checkpoint_mode='{checkpoint_mode}'"
        )
    
    # Find and download checkpoint
    remote_checkpoint = find_latest_checkpoint_path(api, team_id, selected_experiment_task_id, framework_name)
    local_dir = os.path.join(work_dir, 'downloaded_checkpoints')
    local_checkpoint = download_checkpoint_file(api, team_id, remote_checkpoint, local_dir)
    
    # Load metadata
    checkpoint, saved_classes = load_checkpoint_metadata(local_checkpoint)
    logger.info(f"Saved classes in checkpoint: {saved_classes}")
    logger.info(f"Current classes: {current_classes}")
    
    # Finetune mode - flexible class handling
    if checkpoint_mode == "finetune":
        saved_set = set(saved_classes)
        current_set = set(current_classes)
        
        # Case 1: Classes match exactly (possibly different order)
        if saved_set == current_set:
            if saved_classes != current_classes:
                logger.info("The class order is different. Reordering classes")
                class_map = reorder_class_map(saved_classes, project_meta)
            else:
                logger.info("Class names match exactly")
            return local_checkpoint, class_map, None
        
        # Case 2: Same number of classes, different names -> remove head
        elif len(saved_classes) == len(current_classes):
            logger.info("Class names differ but number of classes match. Removing classification head weights")
            modified_checkpoint = remove_classification_head(local_checkpoint)
            return modified_checkpoint, class_map, None
        
        # Case 3: Different number of classes -> keep full checkpoint, ignore head during loading
        else:
            logger.info("Class names and number of classes do not match. Removing classification head weights")
            return local_checkpoint, class_map, None
    
    # Continue mode - strict class matching required
    elif checkpoint_mode == "continue":
        logger.info(f"Continue mode: loading checkpoint from task_id={selected_experiment_task_id}")
        validate_classes_exact_match(saved_classes, current_classes)
        logger.info(f"Checkpoint loaded from task_id={selected_experiment_task_id}")
        state = load_state_json(local_checkpoint)

        return local_checkpoint, class_map, state
    
    else:
        raise ValueError(
            f"Invalid checkpoint_mode='{checkpoint_mode}'. "
            f"Valid values: 'scratch', 'finetune', 'continue'"
        )

def save_state_json(state: dict, checkpoint_path: str):
    """
    Save training state as JSON file next to checkpoint.
    
    Args:
        state: State dict from LiveTraining.state()
        checkpoint_path: Path to .pth checkpoint file
    """
    state_path = checkpoint_path.replace('.pth', '_state.json')
    
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    
    logger.info(f"State saved to {state_path}")


def load_state_json(checkpoint_path: str) -> dict:
    """
    Load training state from JSON file next to checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        
    Returns:
        state: State dict for LiveTraining.load_state()
    """
    state_path = checkpoint_path.replace('.pth', '_state.json')
    
    if not os.path.exists(state_path):
        raise ValueError(
            f"State file not found: {state_path}. "
            f"This checkpoint may not support 'continue' mode."
        )
    
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    logger.info(f"State loaded from {state_path}")
    return state