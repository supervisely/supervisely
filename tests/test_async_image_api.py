import asyncio
import os
import time
from typing import List

from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm

from supervisely import Api, logger
from supervisely._utils import run_coroutine
from supervisely.api.api import ApiField
from supervisely.io.fs import clean_dir, ensure_base_path

# LOG_LEVEL = "INFO"
LOG_LEVEL = "DEBUG"
DATASET_ID = 98357
DATASET_ID = 37
user_path = os.path.expanduser("~")
save_path = f"{user_path}/Work/test_images_download/"
logger.info(f"Save path: {save_path}")
ensure_base_path(save_path)
clean_dir(save_path)
api = Api.from_env()
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
    results = run_coroutine(test_download_np())
    for idx, result in enumerate(results):
        save_image_from_np(result, str(idx), save_path)


async def test_download_path():
    tasks = []
    for image in images:
        path = f"{save_path}{image.name}.png"
        task = api.image.download_path_async(image.id, path)
        tasks.append(task)
    with tqdm(total=len(tasks), desc="Downloading images", unit="image") as pbar:
        start = time.monotonic()
        for f in asyncio.as_completed(tasks):
            _ = await f
            pbar.update(1)
        finish = time.monotonic() - start
        print(f"Time taken: {finish}")


def main_dp():
    run_coroutine(test_download_path())


def main_dps():
    pbar = tqdm(total=len(ids), desc="Downloading images", unit="image")
    run_coroutine(api.image.download_paths_async(ids, paths, progress_cb=pbar))


def compare_main_dps():
    pbar = tqdm(total=len(ids), desc="Downloading images", unit="image")
    start = time.monotonic()
    api.image.download_paths(DATASET_ID, ids, paths, pbar)
    finish = time.monotonic() - start
    print(f"Time taken for bulk method: {finish}")

    start = time.monotonic()
    main_dps()
    finish = time.monotonic() - start
    print(f"Time taken for async method: {finish}")


def main_bytes():
    img_bytes = run_coroutine(api.image.download_bytes_single_async(ids[0]))

    with open(f"{save_path}{ids[0]}.png", "wb") as f:
        f.write(img_bytes)


def main_n_bytes():
    pbar = tqdm(total=len(ids), desc="Downloading images", unit="image")
    img_bytes_list = run_coroutine(api.image.download_bytes_many_async(ids, progress_cb=pbar))
    for img_bytes, name in zip(img_bytes_list, names):
        with open(f"{save_path}{name}", "wb") as f:
            f.write(img_bytes)


async def test_listing():
    start = time.monotonic()
    all_imgs = []
    async for batch in api.image.get_list_generator_async(139, sort_order="desc"):
        all_imgs.extend(batch)
    end = time.monotonic()
    print(f"Time taken for async listing: {end - start}")
    return all_imgs


async def image_get_list_async(
    api: Api,
    project_id: int,
    dataset_id: int = None,
    images_ids: List[int] = None,
    # per_page: int = 500,
):
    method = "images.list"
    data = {
        ApiField.PROJECT_ID: project_id,
        ApiField.FORCE_METADATA_FOR_LINKS: False,
        ApiField.SORT_ORDER: "desc",
        # ApiField.PER_PAGE: per_page,
    }
    if dataset_id is not None:
        data[ApiField.DATASET_ID] = dataset_id
    if images_ids is not None:
        data[ApiField.FILTERS] = [{"field": ApiField.ID, "operator": "in", "value": images_ids}]
    # semaphore = api.get_default_semaphore()
    semaphore = asyncio.Semaphore(5)
    # print("Semaphore init:", semaphore._value)
    pages_count = None
    tasks: List[asyncio.Task] = []

    async def _r(data_, task_i):
        nonlocal pages_count
        async with semaphore:
            # print("Page N:", task_i, "Semaphore:", semaphore._value)
            response = await api.post_async(method, data_)
            response_json = response.json()
            items = response_json.get("entities", [])
            pages_count = response_json["pagesCount"]
            # print(f"Page {task_i} Finshed")
        return [api.image._convert_json_info(item) for item in items]

    data[ApiField.PAGE] = 1
    t = time.monotonic()
    items = await _r(data, 1)
    print(f"Awaited page 1/{pages_count} for {time.monotonic() - t:.4f} sec")
    t = time.monotonic()
    for page_n in range(2, pages_count + 1):
        data[ApiField.PAGE] = page_n
        tasks.append(asyncio.create_task(_r(data.copy(), page_n)))
    for i, task in enumerate(tasks, 2):
        items.extend(await task)
        print(f"awaited page {i}/{pages_count} for {time.monotonic() - t:.4f} sec")
        t = time.monotonic()
    return items


def listing():
    all_imgs = run_coroutine(test_listing())
    return all_imgs


def old_listing():
    start = time.monotonic()
    all_imgs = api.image.get_list(139)
    end = time.monotonic()
    print(f"Time taken for old: {end - start}")
    return all_imgs


def nico_listing():
    start = time.monotonic()
    all_imgs = run_coroutine(image_get_list_async(api, 51, 139))
    end = time.monotonic()
    print(f"Time taken for Nico listing: {end - start}")
    return all_imgs


if __name__ == "__main__":
    try:
        # main_dnp()  # to download and save images as numpy arrays
        # main_dp()  # to download and save images as files
        # main_dps()  # to download and save images as files (batch)
        # compare_main_dps()  # to compare the time taken for downloading images as files (batch)
        # main_bytes()  # to download and save image as bytes
        # main_n_bytes()  # to download and save images as bytes (batch)

        results_1 = listing()
        results_2 = nico_listing()
        assert results_1 == results_2
    except KeyboardInterrupt:
        logger.info("Stopped by user")
