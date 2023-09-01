import numpy as np


def create_img():
    return np.random.randint(0, 255, size=(320, 640, 3))


def compare(img1: np.ndarray, img2: np.ndarray):
    return np.allclose(img1, img2, rtol=1e-16)
