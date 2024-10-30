import asyncio
import time

from tqdm import tqdm
from tqdm.asyncio import tqdm

import supervisely as sly

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
DATASET_ID = 99051
save_path = "/home/ganpoweird/Work/test_video_download/"
sly.fs.ensure_base_path(save_path)
sly.fs.clean_dir(save_path)
api = sly.Api.from_env()
videos = api.video.get_list(DATASET_ID)
ids = [video.id for video in videos]
names = [video.name for video in videos]
paths = [f"{save_path}{video.name}.mp4" for video in videos]

api.logger.setLevel(LOG_LEVEL)


async def test_download_path():
    tasks = []
    for video in videos:
        path = f"{save_path}{video.name}.mp4"
        task = api.video.download_path_async(video.id, path)
        tasks.append(task)
    with tqdm(total=len(tasks), desc="Downloading videos", unit="video") as pbar:
        start = time.time()
        for f in asyncio.as_completed(tasks):
            _ = await f
            pbar.update(1)
        finish = time.time() - start
        print(f"Time taken: {finish}")


def main_dp():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_download_path())


def main_dps():
    start = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        api.video.download_paths_async(ids, paths, show_progress=True, check_hash=False)
    )
    finish = time.time() - start
    print(f"Time taken for async method: {finish}")


def compare_main_dps():
    pbar = tqdm(total=len(ids), desc="Downloading videos", unit="video")
    start = time.time()
    for id, path in zip(ids, paths):
        api.video.download_path(id, path)
        pbar.update(1)
    finish = time.time() - start
    print(f"Time taken for bulk method: {finish}")

    main_dps()


if __name__ == "__main__":
    try:

        # main_dp()  # to download and save videos as files
        main_dps()  # to download and save videos as files (batch)
        # compare_main_dps()  # to compare the time taken for downloading videos as files (batch)
    except KeyboardInterrupt:
        sly.logger.info("Stopped by user")
set().discard
