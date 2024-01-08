import argparse
import json
import os
import sys

from dotenv import load_dotenv

import supervisely as sly

PARENT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LOCAL_ENV = os.path.join(PARENT_PATH, "local.env")
# load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv(os.path.expanduser("~/ninja.env"))

api = sly.Api.from_env()
project_id = 3198

# possible combinations to test
j_settings = [
    {"multiView": {"enabled": False, "tagName": None, "tagId": None, "isSynced": False}},
    {"multiView": {"enabled": True, "tagId": None, "tagName": "im_id", "isSynced": False}},
    {"multiView": {"enabled": True, "tagId": 27855, "tagName": None, "isSynced": True}},
    {
        "multiView": {"enabled": True, "tagId": None, "tagName": "randdddd", "isSynced": False}
    },  # RuntimeError: The multi-view tag 'randdddd' (as planned)
    {
        "multiView": {"enabled": True, "tagId": 99999999, "tagName": None, "isSynced": False}
    },  # RuntimeError: The multi-view tag with ID=99999999 (as planned)
]

for idx, j in enumerate(j_settings):
    meta_json = api.project.get_meta(project_id, with_settings=True)
    meta_json.update({"projectSettings": j})
    m = sly.ProjectMeta.from_json(meta_json)
    s = sly.ProjectSettings.from_json(j)

    api.project.update_meta(project_id, meta_json)
    api.project.update_meta(project_id, m)
    print(idx)

pass
