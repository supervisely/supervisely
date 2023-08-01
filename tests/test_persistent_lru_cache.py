import numpy as np
import shutil
from functools import wraps
from pathlib import Path

from supervisely.nn.inference.cache import PersistentImageLRUCache


def create_img():
    return np.random.randint(0, 255, size=(320, 640, 3))


def clear(tmp):
    shutil.rmtree(tmp)


def clear_callback(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tmp = create_tmp()
        try:
            func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            clear(tmp)

    return wrapper


def create_tmp() -> Path:
    tmp = Path("./tmp").absolute()
    tmp.mkdir(exist_ok=True)
    return tmp


def compare(img1: np.ndarray, img2: np.ndarray):
    return np.allclose(img1, img2, rtol=1e-16)


@clear_callback
def test_save():
    tmp = create_tmp()
    cache = PersistentImageLRUCache(1, tmp)
    img1 = create_img()
    cache[1] = img1

    assert (tmp / "1.png").exists()
    assert compare(cache[1], img1)


@clear_callback
def test_order():
    tmp = create_tmp()
    cache = PersistentImageLRUCache(2, tmp)

    img1, img2, img3 = create_img(), create_img(), create_img()
    cache[1] = img1
    cache[2] = img2
    cache[1]

    cache[3] = img3

    assert 2 not in cache
    assert 1 in cache
    assert 3 in cache
    assert not (tmp / "2.png").exists()


@clear_callback
def test_pop():
    tmp = create_tmp()
    cache = PersistentImageLRUCache(2, tmp)

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
    assert not (tmp / "3.png").exists()


if __name__ == "__main__":
    test_save()
    test_order()
    test_pop()
    print("All test are passed")
