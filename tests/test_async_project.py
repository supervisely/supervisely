import asyncio
import os
import time

import supervisely as sly
from supervisely.project.project import _download_project, _download_project_async

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
PROJECT_ID = 41860
common_path = "/home/ganpoweird/Work/test_project_download/"
save_path = os.path.join(common_path, "old/")
save_path_async = os.path.join(common_path, "async/")
sly.fs.ensure_base_path(common_path)
api = sly.Api.from_env()

sly.fs.clean_dir(common_path)

api.logger.setLevel(LOG_LEVEL)


def main_dpa():
    start = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_download_project_async(api, PROJECT_ID, save_path_async))
    finish = time.time() - start
    print(f"Time taken for async method: {finish}")


def compare_downloads():
    start = time.time()
    _download_project(api, PROJECT_ID, save_path)
    finish = time.time() - start
    print(f"Time taken for old method: {finish}")

    main_dpa()


if __name__ == "__main__":
    try:
        main_dpa()  # to download and save project as files (async)
        # compare_downloads()  # to compare the time taken for downloading and saving project as files (old vs async)
    except KeyboardInterrupt:
        sly.logger.info("Stopped by user")
set().discard
