import os
import pytest
import time
from pathlib import Path

from supervisely.nn.inference.cache import SmartSegCache


# Test methods
def test_load_functions_and_persistance(api_mock, app_mock, tmp_path: Path):
    inf_cache = SmartSegCache(
        app_mock,
        maxsize=10,
        ttl=100,
        base_folder=tmp_path,
    )
    video_id = 1
    frame_indexes = [0, 1, 2, 3]
    sep_frame = 11
    dataset_id = 1
    images = [1, 3, 4, 6]
    sep_img = 2

    existing_imgs = [f"frame_{video_id}_{fi}.png" for fi in frame_indexes]
    existing_imgs.extend([f"image_{imid}.png" for imid in images])
    existing_imgs.append(f"image_{sep_img}.png")
    existing_imgs.append(f"frame_1_{sep_frame}.png")

    inf_cache.download_frames(api_mock, video_id, frame_indexes)
    inf_cache.download_frame(api_mock, video_id, sep_frame)

    inf_cache.download_images(api_mock, dataset_id, images)
    inf_cache.download_image(api_mock, sep_img)

    # Should load images
    assert api_mock.video.frame.download_nps.call_count == 1
    assert sorted(os.listdir(tmp_path)) == sorted(existing_imgs)

    # Should get images from cache
    inf_cache.download_frames(api_mock, video_id, frame_indexes)
    assert api_mock.video.frame.download_nps.call_count == 1


def test_ttl_limit(api_mock, app_mock, tmp_path: Path):
    inf_cache = SmartSegCache(
        app_mock,
        maxsize=10,
        ttl=1,
        base_folder=tmp_path,
    )
    img1, img2 = 1, 2
    existing = [f"image_{img2}.png"]

    # Should remove first image
    inf_cache.download_image(api_mock, img1)
    time.sleep(1)
    inf_cache.download_image(api_mock, img2)

    assert os.listdir(tmp_path) == existing

    # Should reload first image
    inf_cache.download_image(api_mock, img1)
    assert api_mock.image.download_np.call_count == 3
