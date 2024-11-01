import asyncio
import os
import time

import supervisely as sly

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
api = sly.Api.from_env()

api.logger.setLevel(LOG_LEVEL)


TEAM_ID = 567
save_path = "/home/ganpoweird/Work/test_file_download/"
remote_path = "/videos/"
sly.fs.ensure_base_path(save_path)
sly.fs.clean_dir(save_path)
files = (
    ("12391768_3840_2160_30fps.mp4", f"{remote_path}12391768_3840_2160_30fps.mp4"),
    ("MOV_h264.mov", f"{remote_path}MOV_h264.mov"),
    ("MP4_HEVC.mp4", f"{remote_path}MP4_HEVC.mp4"),
)


async def download_files():
    sema = asyncio.Semaphore(10)
    tasks = []
    for name, path in files:
        progress = sly.tqdm_sly(total=None, desc=f"Downloading {name}", unit="B", unit_scale=True)
        task = api.file.download_async(TEAM_ID, path, save_path + name, sema, progress_cb=progress)
        tasks.append(task)
    await asyncio.gather(*tasks)


def main_db():
    loop = asyncio.get_event_loop()
    remote_paths = [path for _, path in files]
    names = [name for name, _ in files]
    save_paths = [save_path + name for name in names]
    progress = sly.tqdm_sly(total=None, desc="Downloading files", unit="B", unit_scale=True)
    loop.run_until_complete(
        api.file.download_bulk_async(TEAM_ID, remote_paths, save_paths, progress_cb=progress)
    )


def maind_df():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_files())


def main_ip():
    start = time.time()
    loop = asyncio.get_event_loop()
    # os.environ["CONTEXT_SLYFILE"] = "/test/test_input.tar"
    os.environ["FOLDER"] = "/videos/"
    os.environ["TEAM_ID"] = str(TEAM_ID)
    loop.run_until_complete(api.file.download_input_async(save_path, show_progress=True))
    end = time.time()
    print(f"Time taken for download input async: {end-start}")


def main_dd():
    start = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(api.file.download_directory_async(TEAM_ID, remote_path, save_path))
    end = time.time()
    print(f"Time taken for download dir async: {end-start}")


def compare_dir_download():
    main_dd()

    sly.fs.clean_dir(save_path)

    start = time.time()
    api.file.download_directory(TEAM_ID, remote_path, save_path)
    end = time.time()
    print(f"Time taken for download dir: {end-start}")


if __name__ == "__main__":
    try:
        # maind_df()  # to download and save files
        # main_dd()  # to download and save files as folder
        # main_db()  # to download and save files in bulk
        # main_ip()  # to download input file
        compare_dir_download()  # to compare download time between async and sync
    except KeyboardInterrupt:
        sly.logger.info("Stopped by user")
set().discard
