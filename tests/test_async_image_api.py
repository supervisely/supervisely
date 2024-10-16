import asyncio
import time

from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm

import supervisely as sly

LOG_LEVEL = "DEBUG"
DATASET_ID = 98357
save_path = "/home/ganpoweird/Work/supervisely/video/images/"

api = sly.Api.from_env()
images = api.image.get_list(DATASET_ID)
api.logger.setLevel(LOG_LEVEL)


def save_image_from_np(img_np, name, save_path):
    img = Image.fromarray(img_np)
    img.save(save_path + name + ".png")


async def test_download_np():
    semaphore = asyncio.Semaphore(10)
    tasks = []
    for image in images:
        task = api.image.async_download_np(image.id, semaphore)
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
    results = [result.result() for result in results]
    for idx, result in enumerate(results):
        save_image_from_np(result, str(idx), save_path)


async def test_download_path():
    semaphore = asyncio.Semaphore(10)
    tasks = []
    for image in images:
        path = f"{save_path}{image.name}.png"
        task = api.image.async_download_path(image.id, path, semaphore)
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


if __name__ == "__main__":
    # main_dnp() # to download and save images as numpy arrays
    main_dp()  # to download and save images as files
