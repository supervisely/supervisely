import os
import shutil
import zipfile

import pytest

from supervisely.convert.converter import ImportManager
from supervisely.io.exception_handlers import ErrorHandler, handle_exception


def _manager() -> ImportManager:
    manager = ImportManager.__new__(ImportManager)
    manager._upload_as_links = False
    return manager


def test_valid_archive_is_unpacked(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("img/1.txt", "content")

    _manager()._unpack_archives(str(tmp_path))

    assert not archive.exists()
    assert (tmp_path / "data" / "img" / "1.txt").read_text() == "content"


def test_corrupted_archive_raises_and_is_reported_as_unpack_error(tmp_path):
    archive = tmp_path / "data.zip"
    archive.write_bytes(os.urandom(1024))

    with pytest.raises(shutil.ReadError) as exc_info:
        _manager()._unpack_archives(str(tmp_path))

    handled = handle_exception(exc_info.value)
    assert isinstance(handled, ErrorHandler.APP.FailedToUnpackArchive)
