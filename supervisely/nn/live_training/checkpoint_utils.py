import os
import re
from typing import Tuple, List, Optional
import json
from supervisely import logger
from supervisely.nn.live_training.helpers import ClassMap
from supervisely.io.json import load_json_file
import supervisely as sly


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

    # Create ClassMap from class names in checkpoint order
    obj_classes = []
    for name in saved_classes:
        obj_class = project_meta.get_obj_class(name)
        if obj_class is None:
            raise ValueError(f"Class '{name}' not found in project metadata")
        obj_classes.append(obj_class)

    return ClassMap(obj_classes)


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
    work_dir: str
) -> Tuple[Optional[str], ClassMap, Optional[dict]]:
    """
    Main orchestrator function to resolve checkpoint loading based on mode.
    
    Args:
        checkpoint_mode: One of 'scratch', 'finetune', 'continue'
        selected_experiment_task_id: Task ID to load checkpoint from (required for finetune/continue)
        class_map: Current ClassMap
        project_meta: Project metadata
        api: Supervisely API instance
        team_id: Team ID
        framework_name: Framework name (unused, kept for compatibility)
        work_dir: Working directory for downloaded files
    
    Returns:
        checkpoint_path: Path to checkpoint file (None for scratch mode)
        class_map: Updated ClassMap (may be reordered for finetune/continue)
        state: Training state dict (only for continue mode)
    """
    checkpoint_name = "latest.pth"
    current_classes = [cls.name for cls in class_map.obj_classes]
    logger.info(f"Checkpoint mode: {checkpoint_mode}")
    
    if checkpoint_mode == "scratch":
        logger.info("Starting from pretrained weights (scratch mode)")
        return None, class_map, None
    
    if selected_experiment_task_id is None:
        raise ValueError(
            f"selected_experiment_task_id must be provided when checkpoint_mode='{checkpoint_mode}'"
        )
    
    # Get experiment info
    task_info = api.task.get_info_by_id(selected_experiment_task_id)
    experiment_info = task_info["meta"]["output"]["experiment"]["data"]

    artifacts_dir = experiment_info["artifacts_dir"]
    model_meta_filename = experiment_info.get("model_meta", "model_meta.json")

    # Setup local paths
    local_dir = os.path.join(work_dir, 'downloaded_checkpoints')
    os.makedirs(local_dir, exist_ok=True)

    # Download checkpoint
    remote_checkpoint = f"{artifacts_dir}checkpoints/{checkpoint_name}"
    local_checkpoint = os.path.join(local_dir, checkpoint_name)
    
    logger.info(f"Downloading checkpoint from {remote_checkpoint}")
    api.file.download(team_id, remote_checkpoint, local_checkpoint)
    logger.info(f"Checkpoint downloaded to {local_checkpoint}")
    
    # Download model_meta.json
    remote_model_meta = f"{artifacts_dir}{model_meta_filename}"
    local_model_meta = os.path.join(local_dir, 'model_meta.json')
    
    logger.info(f"Downloading model_meta from {remote_model_meta}")
    api.file.download(team_id, remote_model_meta, local_model_meta)
    
    # Load saved classes
    model_meta_json = load_json_file(local_model_meta)
    saved_project_meta = sly.ProjectMeta.from_json(model_meta_json)
    saved_classes = [cls.name for cls in saved_project_meta.obj_classes]
    
    logger.info(f"Saved classes: {saved_classes}")
    logger.info(f"Current classes: {current_classes}")
    
    # Finetune mode - flexible class handling
    if checkpoint_mode == "finetune":
        saved_set = set(saved_classes)
        current_set = set(current_classes)
        
        if saved_set == current_set:
            if saved_classes != current_classes:
                logger.info("Class order differs. Reordering classes")
                class_map = reorder_class_map(saved_classes, project_meta)
            else:
                logger.info("Class names match exactly")
            return local_checkpoint, class_map, None
        
        elif len(saved_classes) == len(current_classes):
            # logger.info("Class names differ but count matches. Removing classification head")
            # modified_checkpoint = remove_classification_head(local_checkpoint)
            logger.warning("Class names differ but count matches. Classification head will be kept as is")
            return local_checkpoint, class_map, None
            
        else:
            logger.info("Classes differ completely. Starting from checkpoint")
            return local_checkpoint, class_map, None
    
    # Continue mode - strict matching required
    elif checkpoint_mode == "continue":
        logger.info(f"Continue mode: loading from task_id={selected_experiment_task_id}")
        validate_classes_exact_match(saved_classes, current_classes)

        # Download state JSON
        state_filename = checkpoint_name.replace('.pth', '_state.json')
        remote_state = f"{artifacts_dir}checkpoints/{state_filename}"
        local_state = local_checkpoint.replace('.pth', '_state.json')
        
        logger.info(f"Downloading state from {remote_state}")
        api.file.download(team_id, remote_state, local_state)
        
        # Use existing utility function
        state = load_state_json(local_checkpoint)
        
        logger.info(f"State loaded with {state.get('dataset_size', 0)} samples at iter {state.get('iter', 0)}")
        return local_checkpoint, class_map, state
    
    else:
        raise ValueError(
            f"Invalid checkpoint_mode='{checkpoint_mode}'. Valid: 'scratch', 'finetune', 'continue'"
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