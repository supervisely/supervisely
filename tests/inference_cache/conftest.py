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
    def get_n_frames_gen(vid, fids):
        for fid in fids:
            yield fid, create_img()

    with mock.patch("supervisely.Api"):
        api = sly.Api()
        api.video.frame.download_nps_generator.side_effect = get_n_frames_gen
        api.video.frame.download_np.side_effect = lambda vid, imid: create_img()
        api.image.download_nps_generator.side_effect = get_n_frames_gen
        api.image.download_np.side_effect = lambda im_id: create_img()
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
