import os
import numpy as np
import shutil
from pathlib import Path

from supervisely.nn.inference.cache import PersistentImageLRUCache


def create_img():
    return np.random.randint(0, 255, size=(320, 640, 3))


def clear(tmp):
    shutil.rmtree(tmp)


def create_tmp() -> Path:
    tmp = Path("./tmp").absolute()
    tmp.mkdir(exist_ok=True)
    return tmp


def compare(img1: np.ndarray, img2: np.ndarray):
    return np.allclose(img1, img2, rtol=1e-16)


def test_save():
    tmp = create_tmp()
    cache = PersistentImageLRUCache(1, tmp)
    img1 = create_img()
    cache[1] = img1

    assert (tmp / "1.png").exists()
    assert compare(cache[1], img1)

    clear(tmp)


if __name__ == "__main__":
    test_save()
    print("All test are passed")
