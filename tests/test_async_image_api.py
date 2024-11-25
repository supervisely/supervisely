import asyncio
import os
import time

from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm

import supervisely as sly

# LOG_LEVEL = "INFO"
LOG_LEVEL = "DEBUG"
DATASET_ID = 98357
user_path = os.path.expanduser("~")
save_path = f"{user_path}/Work/test_images_download/"
sly.logger.info(f"Save path: {save_path}")
sly.fs.ensure_base_path(save_path)
sly.fs.clean_dir(save_path)
api = sly.Api.from_env()
images = api.image.get_list(DATASET_ID)
ids = [image.id for image in images]
names = [image.name for image in images]
paths = [f"{save_path}{image.name}.png" for image in images]

api.logger.setLevel(LOG_LEVEL)


def save_image_from_np(img_np, name, save_path):
    img = Image.fromarray(img_np)
    img.save(save_path + name + ".png")


async def test_download_np():
    tasks = []
    for image in images:
        task = api.image.download_np_async(image.id)
        tasks.append(task)
    with tqdm(total=len(tasks), desc="Downloading images", unit="image") as pbar:
        results = []
        for f in asyncio.as_completed(tasks):
            result = await f
            results.append(result)
            pbar.update(1)
    return results


def main_dnp():
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(test_download_np())
    for idx, result in enumerate(results):
        save_image_from_np(result, str(idx), save_path)


async def test_download_path():
    tasks = []
    for image in images:
        path = f"{save_path}{image.name}.png"
        task = api.image.download_path_async(image.id, path)
        tasks.append(task)
    with tqdm(total=len(tasks), desc="Downloading images", unit="image") as pbar:
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
    pbar = tqdm(total=len(ids), desc="Downloading images", unit="image")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(api.image.download_paths_async(ids, paths, progress_cb=pbar))


def compare_main_dps():
    pbar = tqdm(total=len(ids), desc="Downloading images", unit="image")
    start = time.time()
    api.image.download_paths(DATASET_ID, ids, paths, pbar)
    finish = time.time() - start
    print(f"Time taken for bulk method: {finish}")

    start = time.time()
    main_dps()
    finish = time.time() - start
    print(f"Time taken for async method: {finish}")


def main_bytes():
    img_bytes = asyncio.run(api.image.download_bytes_single_async(ids[0]))

    with open(f"{save_path}{ids[0]}.png", "wb") as f:
        f.write(img_bytes)


def main_n_bytes():
    pbar = tqdm(total=len(ids), desc="Downloading images", unit="image")
    loop = asyncio.get_event_loop()
    img_bytes_list = loop.run_until_complete(
        api.image.download_bytes_many_async(ids, progress_cb=pbar)
    )

    for img_bytes, name in zip(img_bytes_list, names):
        with open(f"{save_path}{name}", "wb") as f:
            f.write(img_bytes)


if __name__ == "__main__":
    try:
        # main_dnp()  # to download and save images as numpy arrays
        # main_dp()  # to download and save images as files
        main_dps()  # to download and save images as files (batch)
        # compare_main_dps()  # to compare the time taken for downloading images as files (batch)
        # main_bytes()  # to download and save image as bytes
        # main_n_bytes()  # to download and save images as bytes (batch)        
    except KeyboardInterrupt:
        sly.logger.info("Stopped by user")
