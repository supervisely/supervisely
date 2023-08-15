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


def test_multiple_intersections(tmp_path, api_mock):
    def load_frame(cache: InferenceImageCache, frame: int):
        cache.download_frame(api_mock, 0, frame)

    def load_frames(cache: InferenceImageCache, frames: List[int]):
        cache.download_frames(api_mock, 0, frames)

    cache = InferenceImageCache(100, 60, base_folder=tmp_path)
    sets_and_lists = [
        (False, [1, 3, 5]),
        (False, [2, 4, 6]),
        (True, [1, 1, 2]),
        (True, [3, 4]),
        (True, [5, 5, 1, 6, 7]),
        (False, [3, 5, 7]),
        (True, [5, 3, 6, 7]),
        (False, [1, 2, 3, 4, 5, 6, 7]),
    ]

    set_of_images = set([1, 2, 3, 4, 5, 6, 7])

    threads: List[threading.Thread] = []
    for is_sep, frames in sets_and_lists:
        if is_sep:
            for img_id in frames:
                thread = threading.Thread(target=load_frame, args=(cache, img_id))
                threads.append(thread)
                thread.start()
        else:
            thread = threading.Thread(target=load_frames, args=(cache, frames))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()

    calls = []

    for call in api_mock.video.frame.download_nps.call_args_list:
        calls.extend(call[0][1])

    for call in api_mock.video.frame.download_np.call_args_list:
        calls.append(call[0][1])

    assert set(calls) == set_of_images
