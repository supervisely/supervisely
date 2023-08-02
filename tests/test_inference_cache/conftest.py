import pytest
import mock
from pathlib import Path

import supervisely as sly
from supervisely.nn.inference import cache
from utils import create_img


@pytest.fixture(scope="function")
def tmp_path(tmp_path_factory) -> Path:
    """Temporary path for any data"""
    tmp_dir = tmp_path_factory.mktemp("./tmp")
    return tmp_dir


@pytest.fixture()
def api_mock():
    def get_n_frames(vid, fids):
        if isinstance(fids, int):
            fids = [fids]
        return [create_img() for _ in fids]

    with mock.patch("supervisely.Api"):
        api = sly.Api()
        api.video.frame.download_nps.side_effect = get_n_frames
        yield api


@pytest.fixture()
def app_mock():
    with mock.patch("supervisely.Application"):
        app = sly.Application()
        yield app


@pytest.fixture()
def get_data_dir_mock(monkeypatch, tmp_path):
    get_data_dir = lambda: tmp_path
    monkeypatch.setattr(cache.sly.app, "get_data_dir", get_data_dir)
