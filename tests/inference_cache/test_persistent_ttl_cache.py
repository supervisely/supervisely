import os
import time
from pathlib import Path

import numpy as np

# pylint: disable=import-error
import pytest
from utils import compare, create_img

from supervisely.nn.inference.cache import PersistentImageTTLCache


def test_save(tmp_path: Path):
    cache = PersistentImageTTLCache(1, 10, tmp_path)

    img1 = create_img()
    cache.save_image(1, img1)

    assert (tmp_path / "1.png").exists()
    assert compare(cache.get_image(1), img1)


def test_order(tmp_path: Path):
    cache = PersistentImageTTLCache(2, 10, tmp_path)

    img1, img2, img3 = create_img(), create_img(), create_img()
    cache.save_image(1, img1)
    cache.save_image(2, img2)
    cache[1]

    cache.save_image(3, img3)

    assert 2 not in cache
    assert 1 in cache
    assert 3 in cache
    assert not (tmp_path / "2.png").exists()


def test_pop(tmp_path: Path):
    cache = PersistentImageTTLCache(2, 10, tmp_path)

    img1, img2, img3 = create_img(), create_img(), create_img()
    cache.save_image(1, img1)
    cache.save_image(2, img2)
    cache[1]
    cache.save_image(3, img3)
    cache[1]

    k, v = cache.popitem()
    assert k == 3
    assert 3 not in cache
    assert 1 in cache
    assert not (tmp_path / "3.png").exists()


def test_clear_all(tmp_path: Path):
    cache = PersistentImageTTLCache(3, 10, tmp_path)

    img1, img2, img3 = create_img(), create_img(), create_img()
    cache.save_image(1, img1)
    cache.save_image(2, img2)
    cache.save_image(3, img3)

    cache.clear(rm_base_folder=False)
    assert os.listdir(cache._base_dir) == []

    cache.save_image(1, img1)
    cache.clear(rm_base_folder=True)
    assert not tmp_path.exists()

    cache.save_image(1, img1)
    assert (tmp_path / "1.png").exists()


def test_ttl_limit(tmp_path: Path):
    cache = PersistentImageTTLCache(3, 1, tmp_path)

    img1, img2 = create_img(), create_img()
    cache.save_image(1, img1)
    time.sleep(2)

    cache.save_image(2, img2)

    assert not (tmp_path / "1.png").exists()
    assert (tmp_path / "2.png").exists()

    with pytest.raises(KeyError):
        cache.get_image(1)
