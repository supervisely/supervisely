import asyncio
import os
import time

from tqdm import tqdm
from tqdm.asyncio import tqdm

import supervisely as sly

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
PROJECT_ID = 44090
DATASET_ID = 107523
pcd_ids = [32412155, 32412154, 32412153]  # pointcloud ids 73, 78, 25
user_path = os.path.expanduser("~")
save_path = f"{user_path}/Work/test_pcd_download/"
sly.logger.info(f"Save path: {save_path}")
sly.fs.ensure_base_path(save_path)
sly.fs.clean_dir(save_path)
api = sly.Api.from_env()
datasets = api.dataset.get_list(PROJECT_ID, recursive=True)
pointclouds = []
for dataset in datasets:
    items = api.pointcloud.get_list(dataset.id)
    pointclouds.extend(items)
ids = [pointcloud.id for pointcloud in pointclouds]
names = [pointcloud.name for pointcloud in pointclouds]
paths = [f"{save_path}{pointcloud.name}" for pointcloud in pointclouds]

api.logger.setLevel(LOG_LEVEL)


def main_dps():
    pbar = tqdm(desc="Downloading pointclouds", total=len(ids))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        api.pointcloud.download_paths_async(
            ids,
            paths,
            progress_cb=pbar,
            progress_cb_type="number",
        )
    )


def main_dris():
    r_ids_list = []
    r_pathes_list = []
    for id in ids:
        r_images = api.pointcloud.get_list_related_images(id)
        r_image_ids = [r_image.get("id") for r_image in r_images]
        r_image_paths = [f'{save_path}{id}_{r_image.get("name")}' for r_image in r_images]
        r_ids_list.append(*r_image_ids)
        r_pathes_list.append(*r_image_paths)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(api.pointcloud.download_related_images_async(r_ids_list, r_pathes_list))


def compare_main_dps():
    start = time.time()
    pbar = tqdm(desc="Downloading pointclouds", total=len(ids))
    for id, path in zip(ids, paths):
        api.pointcloud.download_path(id, path)
        pbar.update(1)
    pbar.close()

    finish = time.time() - start
    print(f"Time taken for old method: {finish}")

    sly.fs.clean_dir(save_path)

    start = time.time()
    main_dps()
    finish = time.time() - start
    print(f"Time taken for async method: {finish}")


def main_ann_bulk(ids, pointclouds=None):
    progress = sly.Progress("Downloading annotations", len(ids))
    loop = sly.utils.get_or_create_event_loop()
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            api.pointcloud.annotation.download_bulk_async(
                ids, progress_cb=progress.iters_done_report
            ),
            loop=loop,
        )
        results = future.result()
    else:
        results = loop.run_until_complete(
            api.pointcloud.annotation.download_bulk_async(
                ids, progress_cb=progress.iters_done_report
            )
        )
    return results


if __name__ == "__main__":
    try:
        # main_dps()  # to download and save pointclouds as files (batch)
        # main_dris()  # to download and save related images of pointclouds as files (batch)
        # compare_main_dps()  # to compare the time taken for downloading pointclouds as files (batch)
        start = time.time()
        results = main_ann_bulk(ids)  # to download annotations of pointclouds (batch)
        finish = time.time() - start
    except KeyboardInterrupt:
        sly.logger.info("Stopped by user")
