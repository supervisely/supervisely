import os

import supervisely.convert.image.pdf.pdf_helper as helper
from supervisely import Annotation, ProjectMeta, logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import (
    JUNK_FILES,
    get_file_ext,
    get_file_name,
    mkdir,
    silent_remove,
)
from supervisely.io.json import load_json_file


class PDFConverter(ImageConverter):

    def __str__(self):
        return AvailableImageConverters.PDF

    def validate_format(self) -> bool:
        detected_pdf_cnt = 0
        images_list = []
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if file in JUNK_FILES:
                    continue
                elif ext == ".pdf":
                    images_list.append(full_path)
                    detected_pdf_cnt += 1
                else:
                    continue

        if detected_pdf_cnt == 0:
            return False
        else:
            # create Items
            self._items = []
            for image_path in images_list:
                dir_path, pdf_name = os.path.split(image_path)
                pdf_name_without_ext = get_file_name(pdf_name)
                pdf_dir_path = os.path.join(dir_path, pdf_name_without_ext)
                mkdir(pdf_dir_path)

                # @TODO: add progress for pdf with many pages?
                success = helper.pages_to_images(
                    doc_path=image_path,
                    save_path=pdf_dir_path,
                    dpi=300,
                    logger=logger,
                )
                silent_remove(image_path)

                if not success:
                    continue

                for root, _, files in os.walk(pdf_dir_path):
                    for file in files:
                        image_path = os.path.join(root, file)
                        item = self.Item(image_path)
                        self._items.append(item)
            return True

    def to_supervisely(
        self,
        item: ImageConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Annotation:
        """Convert to Supervisely format."""
        return item.create_empty_annotation()
