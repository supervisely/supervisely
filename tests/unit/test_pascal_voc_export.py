"""Supervisely -> Pascal VOC export (sly.convert.to_pascal_voc, Project.to_pascal_voc).

- Boxes are XML-only: rectangles are not painted into SegmentationClass/SegmentationObject. The
  check used the class's shape, so the rectangles of an AnyShape class were painted as filled
  regions.
- Images that are not JPEG are re-encoded with PIL, which cannot write RGBA (or palette, 16-bit,
  ...) images as JPEG: one RGBA PNG made the whole export fail.
"""

import os

import numpy as np
import pytest
from PIL import Image

import supervisely as sly

pytest.importorskip("lxml", reason="the Pascal VOC export writes its XML with lxml")

IMG_SIZE = (40, 60)  # rows, cols
THING = sly.ObjClass("thing", sly.AnyGeometry, color=[255, 0, 0])
BOX = sly.ObjClass("box", sly.Rectangle, color=[0, 255, 0])
BLOB = sly.ObjClass("blob", sly.Polygon, color=[0, 0, 255])


def _polygon(top, left, bottom, right):
    return sly.Polygon(
        [
            sly.PointLocation(top, left),
            sly.PointLocation(top, right),
            sly.PointLocation(bottom, right),
            sly.PointLocation(bottom, left),
        ]
    )


def _export(tmp_path, items):
    """items: {image name: (PIL image or array, labels)}; returns the VOC root folder."""
    project = sly.Project(str(tmp_path / "project"), sly.OpenMode.CREATE)
    project.set_meta(sly.ProjectMeta(obj_classes=[THING, BOX, BLOB]))
    dataset = project.create_dataset("ds")
    for name, (img, labels) in items.items():
        path = str(tmp_path / name)
        (img if isinstance(img, Image.Image) else Image.fromarray(img)).save(path)
        dataset.add_item_file(name, path, ann=sly.Annotation(img_size=IMG_SIZE, labels=labels))
    project = sly.Project(str(tmp_path / "project"), sly.OpenMode.READ)
    dest = str(tmp_path / "voc")
    project.to_pascal_voc(dest_dir=dest, log_progress=False)
    return os.path.join(dest, "VOCdevkit", "VOC")


def _mask(path):
    return np.array(Image.open(path).convert("RGB"))


def test_rectangles_of_any_shape_classes_are_not_painted_into_masks(tmp_path):
    labels = [
        sly.Label(sly.Rectangle(2, 2, 12, 12), THING),  # AnyShape box
        sly.Label(_polygon(20, 30, 35, 55), THING),  # AnyShape polygon
        sly.Label(sly.Rectangle(2, 30, 12, 40), BOX),  # plain box
        sly.Label(_polygon(20, 2, 35, 20), BLOB),
    ]
    img = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    voc = _export(tmp_path, {"a.jpg": (img, labels)})

    for mask_path in (
        os.path.join(voc, "SegmentationClass", "ds_a_class.png"),
        os.path.join(voc, "SegmentationObject", "ds_a_instance.png"),
    ):
        mask = _mask(mask_path)
        assert not mask[4:11, 4:11].any(), "AnyShape box painted"
        assert not mask[4:11, 32:39].any(), "plain box painted"
        assert mask[27, 42].any(), "AnyShape polygon missing"
        assert mask[27, 10].any(), "polygon missing"

    # the boxes are still objects in the XML
    with open(os.path.join(voc, "Annotations", "ds_a.xml")) as f:
        xml = f.read()
    assert xml.count("<name>thing</name>") == 2
    assert xml.count("<name>box</name>") == 1


@pytest.mark.parametrize("mode", ["RGBA", "P", "LA"])
def test_non_rgb_png_images_are_saved_as_jpeg(tmp_path, mode):
    img = Image.fromarray(np.full((*IMG_SIZE, 3), 200, dtype=np.uint8)).convert(mode)
    voc = _export(
        tmp_path,
        {
            "rgb.png": (np.full((*IMG_SIZE, 3), 100, dtype=np.uint8), []),
            "other.png": (img, [sly.Label(_polygon(5, 5, 20, 20), BLOB)]),
        },
    )

    images = sorted(os.listdir(os.path.join(voc, "JPEGImages")))
    assert images == ["ds_other.jpg", "ds_rgb.jpg"]
    with Image.open(os.path.join(voc, "JPEGImages", "ds_other.jpg")) as exported:
        assert exported.format == "JPEG"
        assert exported.size == (IMG_SIZE[1], IMG_SIZE[0])
