import asyncio
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
        task = api.file.download_async(
            TEAM_ID, path, save_path + name, sema, show_file_progress=True
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


def maind_df():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_files())


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
        main_dd()  # to download and save files as folder
        # compare_dir_download()  # to compare download time between async and sync
    except KeyboardInterrupt:
        sly.logger.info("Stopped by user")
set().discard
