import os
import numpy as np
import shutil
from dotenv import load_dotenv
from pathlib import Path

import supervisely as sly


if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))

tmp_path = Path(".") / "tmp"

api = sly.Api.from_env()

image_id = 19480707
image_info = api.image.get_info_by_id(image_id)

hash_for_loading = image_info.hash
name = image_info.name
path_to_image = tmp_path / f"path_{name}"

api.image.download_paths_by_hashes([hash_for_loading], [path_to_image])
image_with_path = sly.image.read(str(path_to_image))
image_wo_path = api.image.download_nps_by_hashes([hash_for_loading])[0]
sly.image.write(str(tmp_path / f"wo_path_{name}.png"), image_wo_path)

assert np.allclose(image_with_path, image_wo_path)
shutil.rmtree(tmp_path)
