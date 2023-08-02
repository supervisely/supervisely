import os
import numpy as np
import pytest
from pathlib import Path

from supervisely.nn.inference.cache import PersistentImageLRUCache


def create_img():
    return np.random.randint(0, 255, size=(320, 640, 3))


def compare(img1: np.ndarray, img2: np.ndarray):
    return np.allclose(img1, img2, rtol=1e-16)


def test_save(tmp_path: Path):
    cache = PersistentImageLRUCache(1, tmp_path)
    img1 = create_img()
    cache[1] = img1

    assert (tmp_path / "1.png").exists()
    assert compare(cache[1], img1)


def test_order(tmp_path: Path):
    cache = PersistentImageLRUCache(2, tmp_path)

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
    cache = PersistentImageLRUCache(2, tmp_path)

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
    cache = PersistentImageLRUCache(3, tmp_path)

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
