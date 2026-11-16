import sys

import numpy as np
import pytest
from PIL import Image, features

from supervisely.convert.image import image_helper
from supervisely.convert.image.image_helper import EXT_TO_CONVERT, validate_image


def _pixels(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))


def _has_native_avif():
    try:
        return features.check_module("avif")
    except ValueError:
        return False


def _has_pillow_heif():
    try:
        import pillow_heif  # pylint: disable=import-error,unused-import
    except ImportError:
        return False
    return True


needs_avif = pytest.mark.skipif(not _has_native_avif(), reason="Pillow built without AVIF")
needs_heif = pytest.mark.skipif(not _has_pillow_heif(), reason="pillow-heif not installed")


@pytest.fixture
def no_pillow_avif(monkeypatch):
    monkeypatch.setitem(sys.modules, "pillow_avif", None)


@pytest.fixture
def no_pillow_heif(monkeypatch):
    monkeypatch.setitem(sys.modules, "pillow_heif", None)


def _save_heif(img, path):
    from pillow_heif import register_heif_opener  # pylint: disable=import-error

    register_heif_opener()
    img.save(path, format="HEIF")


def test_jfif_keeps_its_bytes_under_a_jpeg_name(tmp_path, no_pillow_avif, no_pillow_heif):
    assert ".jfif" not in EXT_TO_CONVERT
    path = tmp_path / "img.jfif"
    _pixels().save(path, format="JPEG")
    original = path.read_bytes()

    result = validate_image(str(path))

    assert result == str(tmp_path / "img.jpeg")
    assert (tmp_path / "img.jpeg").read_bytes() == original
    assert not path.exists()


def test_jfif_does_not_overwrite_an_existing_jpeg(tmp_path):
    existing = tmp_path / "img.jpeg"
    _pixels(1).save(existing, format="JPEG")
    existing_bytes = existing.read_bytes()
    path = tmp_path / "img.jfif"
    _pixels(2).save(path, format="JPEG")
    jfif_bytes = path.read_bytes()

    result = validate_image(str(path))

    assert result == str(tmp_path / "img.jfif.jpeg")
    assert existing.read_bytes() == existing_bytes
    assert (tmp_path / "img.jfif.jpeg").read_bytes() == jfif_bytes


@needs_avif
def test_avif_converts_without_pillow_avif(tmp_path, no_pillow_avif, no_pillow_heif):
    path = tmp_path / "img.avif"
    _pixels().save(path, format="AVIF")

    result = validate_image(str(path))

    assert result == str(tmp_path / "img.jpeg")
    assert not path.exists()
    with Image.open(result) as img:
        assert img.format == "JPEG"
        assert img.size == (64, 48)


def test_avif_without_any_support_names_what_to_install(monkeypatch, tmp_path, no_pillow_avif):
    monkeypatch.setattr(features, "check_module", lambda name: False)
    path = tmp_path / "img.avif"
    path.write_bytes(b"\x00" * 16)

    with pytest.raises(ImportError, match="pillow-avif-plugin"):
        image_helper.convert_to_jpg(str(path))
    assert path.exists()


@pytest.mark.parametrize("ext", [".heic", ".HEIF"])
def test_heif_without_pillow_heif_is_skipped_with_clear_reason(tmp_path, no_pillow_heif, ext):
    path = tmp_path / f"img{ext}"
    path.write_bytes(b"\x00" * 16)

    with pytest.raises(ImportError, match="pillow-heif"):
        image_helper.convert_to_jpg(str(path))
    assert validate_image(str(path)) is None
    assert path.exists()
    assert list(tmp_path.iterdir()) == [path]


@needs_heif
@pytest.mark.parametrize("ext", [".heic", ".heif"])
def test_heif_converts_with_pillow_heif(tmp_path, no_pillow_avif, ext):
    path = tmp_path / f"img{ext}"
    _save_heif(_pixels(), path)

    result = validate_image(str(path))

    assert result == str(tmp_path / "img.jpeg")
    with Image.open(result) as img:
        assert img.format == "JPEG"
        assert img.size == (64, 48)


@needs_heif
@needs_avif
def test_same_stem_conversions_do_not_overwrite_each_other(tmp_path, no_pillow_avif):
    jpeg = tmp_path / "img.jpeg"
    _pixels(1).save(jpeg, format="JPEG")
    jpeg_bytes = jpeg.read_bytes()
    _pixels(2).save(tmp_path / "img.avif", format="AVIF")
    _save_heif(_pixels(3), tmp_path / "img.heif")

    avif = validate_image(str(tmp_path / "img.avif"))
    heif = validate_image(str(tmp_path / "img.heif"))

    assert avif == str(tmp_path / "img.avif.jpeg")
    assert heif == str(tmp_path / "img.heif.jpeg")
    assert jpeg.read_bytes() == jpeg_bytes
    assert (tmp_path / "img.avif.jpeg").read_bytes() != (tmp_path / "img.heif.jpeg").read_bytes()


@needs_avif
def test_failed_conversion_leaves_no_partial_jpeg(monkeypatch, tmp_path, no_pillow_avif):
    path = tmp_path / "img.avif"
    _pixels().save(path, format="AVIF")

    def broken_save(self, fp, *args, **kwargs):
        with open(fp, "wb") as f:
            f.write(b"\xff\xd8partial")
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", broken_save)

    assert validate_image(str(path)) is None
    assert path.exists()
    assert not (tmp_path / "img.jpeg").exists()
