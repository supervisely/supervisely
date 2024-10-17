import asyncio
import time

from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm

import supervisely as sly

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
DATASET_ID = 98357
save_path = "/home/ganpoweird/Work/supervisely/video/images/"

api = sly.Api.from_env()
images = api.image.get_list(DATASET_ID)
ids = [image.id for image in images]
paths = [f"{save_path}{image.name}.png" for image in images]

api.logger.setLevel(LOG_LEVEL)


def save_image_from_np(img_np, name, save_path):
    img = Image.fromarray(img_np)
    img.save(save_path + name + ".png")


async def test_download_np():
    semaphore = asyncio.Semaphore(10)
    tasks = []
    for image in images:
        task = api.image.download_np_async(image.id, semaphore)
        tasks.append(task)
    with tqdm(total=len(tasks), desc="Downloading images", unit="image") as pbar:
        results = []
        for f in asyncio.as_completed(tasks):
            result = await f
            results.append(result)
            pbar.update(1)
    return results


def main_dnp():
    results = asyncio.run(test_download_np())
    for idx, result in enumerate(results):
        save_image_from_np(result, str(idx), save_path)


async def test_download_path():
    semaphore = asyncio.Semaphore(10)
    tasks = []
    for image in images:
        path = f"{save_path}{image.name}.png"
        task = api.image.download_path_async(image.id, path, semaphore)
        tasks.append(task)
    with tqdm(total=len(tasks), desc="Downloading images", unit="image") as pbar:
        start = time.time()
        for f in asyncio.as_completed(tasks):
            _ = await f
            pbar.update(1)
        finish = time.time() - start
        print(f"Time taken: {finish}")


def main_dp():
    asyncio.run(test_download_path())


def main_dps():
    semaphore = asyncio.Semaphore(33)
    asyncio.run(api.image.download_paths_async(ids, paths, semaphore))


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


if __name__ == "__main__":
    # main_dnp() # to download and save images as numpy arrays
    # main_dp()  # to download and save images as files
    # main_dps()  # to download and save images as files (batch)
    # compare_main_dps()  # to compare the time taken for downloading images as files (batch)
