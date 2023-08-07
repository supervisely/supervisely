import os
from typing import List
import pytest
import threading
from pathlib import Path

from supervisely.nn.inference.cache import InferenceImageCache


# Test methods
def test_multiple_loading(tmp_path, api_mock):
    def load_img(cache: InferenceImageCache, img_id: int):
        cache.download_image(api_mock, img_id)

    def load_frames(cache: InferenceImageCache, frames: List[int]):
        cache.download_frames(api_mock, 0, frames)

    cache = InferenceImageCache(5, 60, base_folder=tmp_path)
    images = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    frames = [[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]]

    set_of_images = set(images)

    threads: List[threading.Thread] = []
    for img_id in images:
        thread = threading.Thread(target=load_img, args=(cache, img_id))
        threads.append(thread)
        thread.start()

    for frame_ids in frames:
        thread = threading.Thread(target=load_frames, args=(cache, frame_ids))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    assert api_mock.image.download_np.call_count == len(set_of_images)
    assert api_mock.video.frame.download_nps.call_args_list[0][0] == (0, [1, 2, 3])
    assert api_mock.video.frame.download_nps.call_args_list[1][0] == (0, [4])
    assert api_mock.video.frame.download_nps.call_count == 2


def test_multiple_deleting(tmp_path, api_mock):
    def clear_cache(cache: InferenceImageCache):
        cache.clear_cache()

    cache = InferenceImageCache(5, 60, base_folder=tmp_path)
    cache.download_image(api_mock, 0)

    threads: List[threading.Thread] = []

    for _ in range(10):
        thread = threading.Thread(target=clear_cache, args=(cache,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
