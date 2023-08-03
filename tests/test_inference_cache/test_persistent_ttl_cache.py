import os
import numpy as np
import pytest
import time
from pathlib import Path

from supervisely.nn.inference.cache import PersistentImageTTLCache
from utils import compare, create_img


def test_save(tmp_path: Path):
    cache = PersistentImageTTLCache(1, 10, tmp_path)

    img1 = create_img()
    cache[1] = img1

    assert (tmp_path / "1.png").exists()
    assert compare(cache[1], img1)


def test_order(tmp_path: Path):
    cache = PersistentImageTTLCache(2, 10, tmp_path)

    img1, img2, img3 = create_img(), create_img(), create_img()
    cache[1] = img1
    cache[2] = img2
    cache[1]

    cache[3] = img3

    assert 2 not in cache
    assert 1 in cache
    assert 3 in cache
    assert not (tmp_path / "2.png").exists()


def test_pop(tmp_path: Path):
    cache = PersistentImageTTLCache(2, 10, tmp_path)

    img1, img2, img3 = create_img(), create_img(), create_img()
    cache[1] = img1
    cache[2] = img2
    cache[1]
    cache[3] = img3
    cache[1]

    k, v = cache.popitem()
    assert k == 3
    assert compare(v, img3)
    assert 3 not in cache
    assert 1 in cache
    assert not (tmp_path / "3.png").exists()


def test_clear_all(tmp_path: Path):
    cache = PersistentImageTTLCache(3, 10, tmp_path)

    img1, img2, img3 = create_img(), create_img(), create_img()
    cache[1] = img1
    cache[2] = img2
    cache[3] = img3

    cache.clear(rm_base_folder=False)
    assert os.listdir(cache._base_dir) == []

    cache[1] = img1
    cache.clear(rm_base_folder=True)
    assert not tmp_path.exists()

    cache[1] = img1
    assert (tmp_path / "1.png").exists()


def test_ttl_limit(tmp_path: Path):
    cache = PersistentImageTTLCache(3, 1, tmp_path)

    img1, img2 = create_img(), create_img()
    cache[1] = img1
    time.sleep(2)

    cache[2] = img2

    assert not (tmp_path / "1.png").exists()
    assert (tmp_path / "2.png").exists()

    with pytest.raises(KeyError):
        cache[1]
