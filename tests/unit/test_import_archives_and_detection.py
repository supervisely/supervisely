import os
import subprocess
import sys
import textwrap
import zipfile

import numpy as np
import pytest
from PIL import Image

import supervisely.convert.converter as converter_module
from supervisely.convert.converter import ImportManager
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.image_helper import validate_mimetypes
from supervisely.io.exception_handlers import ErrorHandler, handle_exception
from supervisely.project.project_type import ProjectType


def _write_image(path, ext="png"):
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(img).save(path, format="JPEG" if ext in ("jpg", "jpeg") else ext.upper())


@pytest.fixture
def import_env(tmp_path, monkeypatch):
    """ImportManager on a local input dir, with no Supervisely instance behind it."""
    monkeypatch.setenv("SLY_APP_DATA_DIR", str(tmp_path / "app_data"))
    monkeypatch.setenv("TEAM_ID", "1")
    monkeypatch.setattr(converter_module.Api, "from_env", classmethod(lambda cls: object()))
    warnings = []
    original_warning = converter_module.logger.warning

    def capture_warning(msg, *args, **kwargs):
        warnings.append(str(msg))
        return original_warning(msg, *args, **kwargs)

    monkeypatch.setattr(converter_module.logger, "warning", capture_warning)
    src = tmp_path / "input"
    src.mkdir()
    return src, warnings


def test_is_image_reads_only_the_header_of_a_large_file(tmp_path):
    """A large non-image file is sniffed at constant memory, not read whole."""
    if not sys.platform.startswith("linux"):
        pytest.skip("ru_maxrss is in KiB on Linux only")
    big = tmp_path / "big.zip"
    with open(big, "wb") as f:
        f.truncate(256 * 1024 * 1024)  # sparse: no disk used, but f.read() would allocate it all

    script = textwrap.dedent(
        f"""
        import resource
        from supervisely.convert.image.image_converter import ImageConverter
        from supervisely.convert.image.image_helper import validate_mimetypes
        conv = ImageConverter({str(tmp_path)!r})
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert conv.is_image({str(big)!r}) is False
        validate_mimetypes("big.zip", {str(big)!r})
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(after - before)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    grown_kib = int(out.stdout.strip().splitlines()[-1])
    assert grown_kib < 16 * 1024, f"peak RSS grew by {grown_kib} KiB"


def test_is_image_follows_symlinks(tmp_path):
    real = tmp_path / "real.png"
    _write_image(real)
    link = tmp_path / "link.png"
    link.symlink_to(real)
    conv = ImageConverter(str(tmp_path))
    assert conv.is_image(str(real)) is True
    assert conv.is_image(str(link)) is True
    assert validate_mimetypes("link.png", str(link)) == "link.png"


def test_validate_mimetypes_renames_by_content(tmp_path):
    path = tmp_path / "photo.png"
    _write_image(path, ext="jpg")  # JPEG bytes under a .png name
    assert validate_mimetypes("photo.png", str(path)) == "photo.jpg"


def test_only_a_corrupt_zip_fails_with_unpack_error_naming_it(import_env):
    src, _ = import_env
    (src / "supervisely_import.zip").write_bytes(os.urandom(64 * 1024))

    with pytest.raises(Exception) as exc_info:
        ImportManager(str(src), ProjectType.IMAGES)

    handled = handle_exception(exc_info.value)
    assert isinstance(handled, ErrorHandler.APP.FailedToUnpackArchive)
    assert handled.code == 1006
    assert "supervisely_import.zip" in handled.message
    assert "is not a zip file" in handled.message
    assert "Upload the archive again" in handled.message
    assert "app_data" not in handled.message  # no container paths in the user-facing text


def test_only_a_7z_fails_with_unsupported_format(import_env):
    src, _ = import_env
    (src / "data.7z").write_bytes(b"7z\xbc\xaf\x27\x1c" + os.urandom(1024))

    with pytest.raises(Exception) as exc_info:
        ImportManager(str(src), ProjectType.IMAGES)

    handled = handle_exception(exc_info.value)
    assert isinstance(handled, ErrorHandler.APP.FailedToUnpackArchive)
    assert "'data.7z' is not a supported archive format" in handled.message
    assert "zip or tar" in handled.message


def test_corrupt_zip_next_to_images_imports_images_and_warns(import_env):
    src, warnings = import_env
    for i in range(3):
        _write_image(src / f"{i}.png")
    (src / "broken.zip").write_bytes(os.urandom(64 * 1024))

    manager = ImportManager(str(src), ProjectType.IMAGES)

    names = sorted(item.name for item in manager.get_items())
    assert names == ["0.png", "1.png", "2.png"]
    assert any("broken.zip" in w and "corrupt or incomplete" in w for w in warnings)


def test_7z_next_to_images_imports_as_before_and_warns(import_env):
    src, warnings = import_env
    for i in range(2):
        _write_image(src / f"{i}.jpg", ext="jpg")
    (src / "extra.7z").write_bytes(b"7z\xbc\xaf\x27\x1c" + os.urandom(1024))

    manager = ImportManager(str(src), ProjectType.IMAGES)

    names = sorted(item.name for item in manager.get_items())
    assert names == ["0.jpg", "1.jpg"]
    assert any("extra.7z" in w and "not a supported archive format" in w for w in warnings)


def test_valid_zip_is_unpacked_without_warning(import_env, tmp_path):
    src, warnings = import_env

    img = tmp_path / "a.png"
    _write_image(img)
    with zipfile.ZipFile(src / "data.zip", "w") as zf:
        zf.write(img, "data/a.png")

    manager = ImportManager(str(src), ProjectType.IMAGES)

    assert [item.name for item in manager.get_items()] == ["a.png"]
    assert not any("could not be unpacked" in w for w in warnings)


def test_unpack_failure_that_is_not_corruption_is_not_called_corrupt(import_env, monkeypatch):
    src, _ = import_env
    with zipfile.ZipFile(src / "data.zip", "w") as zf:
        zf.writestr("a.txt", "x")

    def no_space(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(converter_module, "unpack_archive", no_space)
    with pytest.raises(Exception) as exc_info:
        ImportManager(str(src), ProjectType.IMAGES)

    handled = handle_exception(exc_info.value)
    assert isinstance(handled, ErrorHandler.APP.FailedToUnpackArchive)
    assert "'data.zip' could not be unpacked" in handled.message
    assert "No space left on device" in handled.message
    assert "corrupt" not in handled.message
