import asyncio
import os
import time

from tqdm import tqdm

from supervisely import Api, logger
from supervisely._utils import run_coroutine
from supervisely.io.fs import clean_dir, ensure_base_path
from supervisely.project.download import download_fast, download_to_cache
from supervisely.project.project import _download_project, _download_project_async

# LOG_LEVEL = "INFO"
LOG_LEVEL = "DEBUG"
PROJECT_ID = 11
DATSET_ID = 12
home_dir = os.path.expanduser("~")
common_path = os.path.join(home_dir, "test_project_download/")
save_path = os.path.join(common_path, "old/")
save_path_async = os.path.join(common_path, "async/")
ensure_base_path(common_path)
api = Api.from_env()

clean_dir(common_path)


def main_df():
    download_fast(api, PROJECT_ID, save_path_async)


def main_dpa():
    coroutine = _download_project_async(api, PROJECT_ID, save_path_async, resume_download=True)
    start = time.monotonic()
    run_coroutine(coroutine)
    finish = time.monotonic() - start
    print(f"Time taken for async method: {finish}")
    print(f"Project downloaded to {save_path_async}")


def main_dptc():
    start = time.monotonic()
    download_to_cache(api, PROJECT_ID)
    finish = time.monotonic() - start
    print(f"Time taken for cache method: {finish}")


def main_dps():
    start = time.monotonic()
    _download_project(api, PROJECT_ID, save_path)
    finish = time.monotonic() - start
    print(f"Time taken for sync method: {finish}")


def compare_downloads():
    main_dps()
    clean_dir(save_path)
    main_dpa()
    clean_dir(save_path_async)


def ann_db():
    imgs = api.image.get_list(DATSET_ID)
    img_ids = [img.id for img in imgs]
    pbar = tqdm(desc="Downloading annotations", unit="B", unit_scale=True)
    coroutne = api.annotation.download_batch_async(
        DATSET_ID,
        img_ids,
        semaphore=asyncio.Semaphore(200),
        progress_cb=pbar,
        progress_cb_type="size",
        # force_metadata_for_links=False,
    )
    start = time.monotonic()
    anns = run_coroutine(coroutne)
    finish = time.monotonic() - start
    print(f"Time taken to download {len(anns)} annotations: {finish}")


if __name__ == "__main__":
    try:
        # api.logger.setLevel(LOG_LEVEL)
        # ann_db()
        # main_dpa()  # to download and save project as files (async)
        # main_dps()  # to download and save project as files (sync)
        # compare_downloads()  # to compare the time taken for downloading and saving project as files (sync vs async)
        main_df()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
