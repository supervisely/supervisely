from supervisely import Annotation, ProjectMeta
from supervisely.convert.base_converter import AvailableImageFormats, BaseConverter
from supervisely.io.fs import get_file_ext, list_files_recursively
from supervisely.io.json import load_json_file

SLY_ANN_KEYS = ["imageName", "imageId", "createdAt", "updatedAt", "annotation"]

class SLYImageConverter(BaseConverter):
    def __init__(self, input_data):
        super().__init__(input_data)

    def __str__(self):
        return AvailableImageFormats.SLY

    def validate_ann_file(self, ann_path):
        if self.meta is None:
            pass
        pass

    def require_key_file(self):
        return True
    
    def validate_key_files(self):
        jsons = list_files_recursively(self.items, valid_extensions=[".json"])
        # TODO: find meta.json first
        for key_file in jsons:
            try:
                self.meta = ProjectMeta.from_json(load_json_file(key_file))
                return True
            except Exception:
                continue

    @property
    def ann_ext(self):
        return ".json" 

    def require_key_file(self):
        return False

    def get_meta(self):
        return self.meta
    
    def get_items(self): # -> generator?
        raise NotImplementedError()

    def to_supervisely(self, item_path: str, ann_path: str) -> Annotation:
        """Convert to Supervisely format."""

        if self.meta is None:
            self.meta = self.get_meta()
        raise NotImplementedError()
