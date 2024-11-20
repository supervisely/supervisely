import argparse
import asyncio
import os
import signal
import sys
import time

import supervisely as sly
from supervisely.project.download import download_to_cache
from supervisely.project.project import _download_project, _download_project_async

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
PROJECT_ID = 325865  #  41862
DATSET_ID = 98429
home_dir = os.path.expanduser("~")
common_path = os.path.join(home_dir, "test_project_download/")
save_path = os.path.join(common_path, "old/")
save_path_async = os.path.join(common_path, "async/")
sly.fs.ensure_base_path(common_path)
api = sly.Api.from_env()

# sly.fs.clean_dir(common_path)


def main_dpa(project_id: int, semaphore_size: int):
    if semaphore_size is None:
        sema = None
    else:
        sema = asyncio.Semaphore(semaphore_size)
    start = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        _download_project_async(
            api, project_id, save_path_async, semaphore=sema, resume_download=True
        )
    )
    finish = time.time() - start
    print(f"Time taken for async method: {finish}")
    print(f"Project downloaded to {save_path_async}")


def main_dptc():
    start = time.time()
    download_to_cache(api, PROJECT_ID)
    finish = time.time() - start
    print(f"Time taken for cache method: {finish}")


def main_dps(project_id: int):
    start = time.time()
    _download_project(api, project_id, save_path)
    finish = time.time() - start
    print(f"Time taken for sync method: {finish}")


def compare_downloads(project_id: int, semaphore_size: int):
    main_dps(project_id)
    sly.fs.clean_dir(save_path)
    main_dpa(project_id, semaphore_size)
    sly.fs.clean_dir(save_path_async)


def ann_db():
    imgs = api.image.get_list(DATSET_ID)
    img_ids = [img.id for img in imgs]
    pbar = sly.tqdm_sly(desc="Downloading annotations", unit="B", unit_scale=True)
    loop = asyncio.get_event_loop()
    anns = loop.run_until_complete(
        api.annotation.download_batch_async(
            DATSET_ID,
            img_ids,
            semaphore=asyncio.Semaphore(200),
            progress_cb=pbar,
            progress_cb_type="size",
            # force_metadata_for_links=False,
        )
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Download and save project as files (async)")
    # parser.add_argument(
    #     "--server", type=str, default=os.environ["SERVER_ADDRESS"], help="Server address"
    # )
    # parser.add_argument("--token", type=str, default=os.environ["API_TOKEN"], help="API token")
    # parser.add_argument("--id", type=int, default=PROJECT_ID, help="ID of the project to download")
    # parser.add_argument(
    #     "--semaphore", type=int, default=50, help="Semaphore size for async download"
    # )
    # args = parser.parse_args()
    try:
        # ann_db()
        # api = sly.Api(args.server, args.token)
        # api.logger.setLevel(LOG_LEVEL)
        main_dpa(PROJECT_ID, 200)  # to download and save project as files (async)
        # main_dps(args.id)  # to download and save project as files (sync)
        # compare_downloads(args.id, args.semaphore)  # to compare the time taken for downloading and saving project as files (sync vs async)
    except KeyboardInterrupt:
        sly.logger.info("Stopped by user")
