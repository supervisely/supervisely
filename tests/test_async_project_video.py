import argparse
import asyncio
import os
import signal
import sys
import time

import supervisely as sly

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
PROJECT_ID = 42112  #  41862
DATSET_ID = 98429
home_dir = os.path.expanduser("~")
common_path = os.path.join(home_dir, "test_project_download/")
save_path = os.path.join(common_path, "sync/")
save_path_async = os.path.join(common_path, "async/")
sly.fs.ensure_base_path(common_path)
api = sly.Api.from_env()

sly.fs.clean_dir(common_path)


def main_dpa(project_id: int, semaphore_size: int):
    sema = asyncio.Semaphore(semaphore_size)
    start = time.time()
    loop = sly.utils.get_or_create_event_loop()
    loop.run_until_complete(
        sly.VideoProject.download_async(api, project_id, save_path_async, semaphore=sema)
    )
    finish = time.time() - start
    print(f"Time taken for async method: {finish}")
    print(f"Project downloaded to {save_path_async}")


def main_dps(project_id: int):
    start = time.time()
    sly.VideoProject.download(api, project_id, save_path)
    finish = time.time() - start
    print(f"Time taken for old method: {finish}")


def compare_downloads(project_id: int, semaphore_size: int):
    main_dps(project_id)
    sly.fs.clean_dir(save_path)
    main_dpa(project_id, semaphore_size)
    sly.fs.clean_dir(save_path_async)


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
        # main_dpa(PROJECT_ID, 200)  # to download and save project as files (async)
        # main_dps(args.project_id)  # to download and save project as files (sync)
        compare_downloads(
            PROJECT_ID, 200
        )  # to compare the time taken for downloading and saving project as files (sync vs async)
    except KeyboardInterrupt:
        sly.logger.info("Stopped by user")
