import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def find_requirements(src_path):
    p = Path(src_path) / "requirements.txt"
    if p.exists():
        return str(p)
    p = Path("requirements.txt")
    if p.exists():
        return str(p)
    return None

def get_config():
    with open("config.json") as f:
        return json.load(f)

def render():
    config = get_config()
    print("Rendering web app with config:", config)
    src_dir = config["src_dir"]
    gui_dir = config["gui_folder_path"]
    main_script = config["main_script"]

    # to be able to import sly_sdk in main
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    # import main
    module_name = Path(main_script).stem
    spec = importlib.util.spec_from_file_location(module_name, main_script)
    main = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = main
    spec.loader.exec_module(main)

    main.app.render(
        main_script_path=main_script,
        src_dir=src_dir,
        app_dir=gui_dir,
        requirements_path=find_requirements(src_dir),
    )

    return gui_dir

if __name__ == "__main__":
    render()
