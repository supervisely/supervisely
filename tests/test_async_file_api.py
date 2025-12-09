import asyncio
import os
import time

from tqdm import tqdm

from supervisely import Api, logger
from supervisely._utils import run_coroutine
from supervisely.io.fs import clean_dir, ensure_base_path

# LOG_LEVEL = "INFO"
LOG_LEVEL = "DEBUG"
api = Api.from_env()

api.logger.setLevel(LOG_LEVEL)

TEAM_ID = 10
user_path = os.path.expanduser("~")
source_path = f"{user_path}/test_project_download/"
save_path = f"{user_path}/Work/test_project_download/"
logger.info(f"Save path: {save_path}")
remote_path = "/videos/"
remote_upload_test_sync = "/test_sync/"
remote_upload_test_async = "/test_async/"
ensure_base_path(save_path)
clean_dir(save_path)
files = (
    ("12391768_3840_2160_30fps.mp4", f"{remote_path}12391768_3840_2160_30fps.mp4"),
    ("MOV_h264.mov", f"{remote_path}MOV_h264.mov"),
    ("MP4_HEVC.mp4", f"{remote_path}MP4_HEVC.mp4"),
)
file_name = "vids.tar"


async def download_files():
    sema = asyncio.Semaphore(10)
    tasks = []
    for name, path in files:
        progress = tqdm(total=None, desc=f"Downloading {name}", unit="B", unit_scale=True)
        task = api.file.download_async(TEAM_ID, path, save_path + name, sema, progress_cb=progress)
        tasks.append(task)
    await asyncio.gather(*tasks)


def main_db():
    remote_paths = [path for _, path in files]
    names = [name for name, _ in files]
    sizeb_list = [api.file.get_info_by_path(TEAM_ID, path).sizeb for _, path in files]
    sizeb = sum(sizeb_list)
    save_paths = [save_path + name for name in names]
    progress = tqdm(total=sizeb, desc="Downloading files", unit="B", unit_scale=True)
    download_coro = api.file.download_bulk_async(
        TEAM_ID, remote_paths, save_paths, progress_cb=progress
    )
    run_coroutine(download_coro)


def maind_df():
    run_coroutine(download_files())


def main_ip():
    start = time.time()
    # os.environ["CONTEXT_SLYFILE"] = "/test/test_input.tar"
    os.environ["FOLDER"] = "/videos/"
    os.environ["TEAM_ID"] = str(TEAM_ID)
    download_coro = api.file.download_input_async(save_path, show_progress=True)
    run_coroutine(download_coro)
    end = time.time()
    print(f"Time taken for download input async: {end-start}")


def main_dd():
    start = time.time()
    download_coro = api.file.download_directory_async(
        TEAM_ID, remote_path, save_path, semaphore=asyncio.Semaphore(100)
    )
    run_coroutine(download_coro)
    end = time.time()
    print(f"Time taken for download dir async: {end-start}")


def compare_dir_download():
    main_dd()

    clean_dir(save_path)

    start = time.time()
    api.file.download_directory(TEAM_ID, remote_path, save_path)
    end = time.time()
    print(f"Time taken for download dir: {end-start}")


def upload():
    total = os.path.getsize(source_path + file_name)
    process_cb = tqdm(total=total, desc="Uploading files", unit="B", unit_scale=True)
    upload_coro = api.file.upload_async(
        TEAM_ID,
        source_path + file_name,
        remote_path + file_name,
        progress_cb=process_cb,
    )
    start = time.monotonic()
    run_coroutine(upload_coro)
    end = time.monotonic()
    logger.info(f"Time taken for upload async: {(end-start) / 60}")


def upload_bulk():
    total = 0
    src_paths = []
    dst_paths = []
    for file in os.listdir(source_path):
        path = os.path.join(source_path, file)
        src_paths.append(path)
        dst_paths.append(remote_path + file)
        total += os.path.getsize(path)
    process_cb = tqdm(total=total, desc="Uploading files", unit="B", unit_scale=True)
    upload_coro = api.file.upload_bulk_async(
        TEAM_ID,
        src_paths,
        dst_paths,
        progress_cb=process_cb,
    )
    start = time.monotonic()
    run_coroutine(upload_coro)
    end = time.monotonic()
    logger.info(f"Time taken for upload bulk async: {(end-start) / 60}")


def main_ud():
    total = 0
    for file in os.listdir(source_path):
        path = os.path.join(source_path, file)
        total += os.path.getsize(path)

    process_cb = tqdm(total=total, desc="Uploading files", unit="B", unit_scale=True)
    start = time.monotonic()
    api.file.upload_directory(
        TEAM_ID,
        source_path,
        remote_upload_test_sync,
        progress_size_cb=process_cb,
    )
    end = time.monotonic()
    logger.info(f"Time taken for upload dir: {end-start}")

    process_cb_as = tqdm(total=total, desc="Uploading files async", unit="B", unit_scale=True)
    coro = api.file.upload_directory_async(
        TEAM_ID,
        source_path,
        remote_upload_test_async,
        progress_size_cb=process_cb_as,
    )
    start = time.monotonic()
    run_coroutine(coro)
    end = time.monotonic()
    logger.info(f"Time taken for upload dir async: {end-start}")


if __name__ == "__main__":
    try:
        # maind_df()  # to download and save files
        # main_dd()  # to download and save files as folder
        # main_db()  # to download and save files in bulk
        # main_ip()  # to download input file
        # compare_dir_download()  # to compare download time between async and sync
        main_ud()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
