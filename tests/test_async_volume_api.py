import asyncio
import os
import time

from tqdm import tqdm
from tqdm.asyncio import tqdm

import supervisely as sly

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
PROJECT_ID = 42569
user_path = os.path.expanduser("~")
save_path = f"{user_path}/Work/test_volumes_download/"
sly.logger.info(f"Save path: {save_path}")
sly.fs.ensure_base_path(save_path)
sly.fs.clean_dir(save_path)
api = sly.Api.from_env()
datasets = api.dataset.get_list(PROJECT_ID, recursive=True)
volumes = []
for dataset in datasets:
    items = api.volume.get_list(dataset.id)
    volumes.extend(items)
ids = [volume.id for volume in volumes]
names = [volume.name for volume in volumes]
paths = [f"{save_path}{volume.name}" for volume in volumes]

api.logger.setLevel(LOG_LEVEL)


def main_dps():
    pbar = sly.tqdm_sly(total=len(ids), desc="Downloading volumes", unit="volume")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(api.volume.download_paths_async(ids, paths, progress_cb=pbar))


def compare_main_dps():
    start = time.time()
    for id, path in zip(ids, paths):
        pbar = tqdm(desc="Downloading volume", unit="B", unit_scale=True)
        api.volume.download_path(id, path, pbar)
        pbar.close()

    finish = time.time() - start
    print(f"Time taken for old method: {finish}")

    start = time.time()
    main_dps()
    finish = time.time() - start
    print(f"Time taken for async method: {finish}")


if __name__ == "__main__":
    try:
        # main_dps()  # to download and save volumes as files (batch)
        compare_main_dps()  # to compare the time taken for downloading volumes as files (batch)

    except KeyboardInterrupt:
        sly.logger.info("Stopped by user")
