from typing import Any, Dict, List, Union

import supervisely.io.fs as sly_fs
from supervisely.io.json import load_json_file


def load_file(file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if not isinstance(file_path, str):
        raise ValueError("Provide a path to a '.json' or '.yaml' file.")

    if not sly_fs.file_exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = sly_fs.get_file_ext(file_path).lower()
    if ext == ".json":
        return load_json_file(file_path)
    elif ext in [".yaml", ".yml"]:
        with open(file_path, "r") as f:
            return f.read()
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Provide a path to a '.json' or '.yaml' file."
        )


def validate_list_of_dicts(data: List[Dict], name: str) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list of dicts, or a path to a '.json' file.")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Each item in {name} must be a dict.")
    return data
