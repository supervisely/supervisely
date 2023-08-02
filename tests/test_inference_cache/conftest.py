import pytest
from pathlib import Path


@pytest.fixture(scope="function")
def tmp_path(tmp_path_factory) -> Path:
    """Temporary path for any data"""
    tmp_dir = tmp_path_factory.mktemp("./tmp")
    return tmp_dir
