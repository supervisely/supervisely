from types import SimpleNamespace

from PIL import Image

from supervisely.annotation.annotation import Annotation
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.imaging.image import get_image_shape
from supervisely.project.project_meta import ProjectMeta


def _save_png_with_orientation(path, size=(3, 7), orientation=6):
    image = Image.new("RGBA", size, (255, 0, 0, 255))
    exif = Image.Exif()
    exif[274] = orientation
    image.save(path, exif=exif)


def test_image_shape_respects_transposed_exif_orientation(tmp_path):
    image_path = tmp_path / "oriented.png"
    _save_png_with_orientation(image_path)

    with Image.open(image_path) as image:
        assert image.size == (3, 7)
        assert image.getexif().get(274) == 6

    assert get_image_shape(str(image_path)) == (3, 7)
    assert Annotation.from_img_path(str(image_path)).img_size == (3, 7)

    item = ImageConverter.Item(str(image_path))
    item.set_shape()
    assert item.shape == (3, 7)


def test_image_upload_uses_returned_server_shape_for_annotations(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (3, 7), (255, 0, 0)).save(image_path)

    converter = ImageConverter(str(tmp_path))
    converter._items = [ImageConverter.Item(str(image_path))]

    class DatasetApi:
        def get_info_by_id(self, dataset_id, raise_error=False):
            return SimpleNamespace(id=dataset_id, name="dataset", project_id=10)

    class ProjectApi:
        def get_meta(self, project_id, with_settings=False):
            return ProjectMeta().to_json()

    class ImageApi:
        def get_list(self, dataset_id):
            return []

        def upload_paths(self, *args, **kwargs):
            return [SimpleNamespace(id=123, height=11, width=22)]

    class AnnotationApi:
        def __init__(self):
            self.uploaded_anns = None

        def upload_anns(self, img_ids, anns, skip_bounds_validation=False):
            self.uploaded_anns = anns

    fake_api = SimpleNamespace(
        dataset=DatasetApi(),
        project=ProjectApi(),
        image=ImageApi(),
        annotation=AnnotationApi(),
        optimization_context={},
    )

    converter.upload_dataset(fake_api, dataset_id=1, log_progress=False)

    assert fake_api.annotation.uploaded_anns[0].img_size == (11, 22)
