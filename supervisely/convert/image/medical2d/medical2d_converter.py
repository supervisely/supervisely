import os
from typing import List, Union

import magic
import nrrd
import numpy as np

from supervisely import Annotation, Api, batched, generate_free_name, is_development, ProjectMeta, logger, TagMeta, Tag, TagValueType
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.medical2d import medical2d_helper as helper
from supervisely.io.fs import remove_dir, get_file_ext, mkdir, get_file_name
from supervisely.volume.volume import is_nifti_file


# @TODO: add group tags?
class Medical2DImageConverter(ImageConverter):

    def __init__(self, input_data: str, labeling_interface: str) -> None:
        self._input_data: str = input_data
        self._items: List[ImageConverter.Item] = []
        self._meta: ProjectMeta = None
        self._labeling_interface = labeling_interface
        self._filtered = None
        self._group_tag_names = []

    def __str__(self):
        return AvailableImageConverters.MEDICAL2D

    def validate_labeling_interface(self) -> bool:
        """Only medical labeling interface can be used for medical images."""
        return self._labeling_interface in ["default", "medical_imaging_single", "images_with_16_color"]

    def validate_format(self) -> bool:
        logger.debug(f"Validating format: {self.__str__()}")

        converted_dir_name = "converted"
        converted_dir = os.path.join(self._input_data, converted_dir_name)
        mkdir(converted_dir, remove_content_if_exists=True)
        meta = ProjectMeta()

        nrrd = {}
        for root, _, files in os.walk(self._input_data):
            if converted_dir_name in root:
                continue
            for file in files:
                path = os.path.join(root, file)
                ext = get_file_ext(path).lower()
                mime = magic.from_file(path, mime=True)
                if mime == "application/dicom":
                    if helper.is_dicom_file(path):  # is dicom
                        paths, names, group_tag = helper.convert_dcm_to_nrrd(path, converted_dir)
                        if group_tag is not None:
                            tag_name = group_tag["name"]
                            tag_value = group_tag["value"]
                            tag_meta = meta.get_tag_meta(tag_name)
                            if tag_meta is None:
                                tag_meta = TagMeta(tag_name, TagValueType.ANY_STRING)
                                meta = meta.add_tag_meta(tag_meta)
                            group_tag = Tag(tag_meta, tag_value)
                        for path, name in zip(paths, names):
                            nrrd[name] = (path, group_tag)
                elif ext == ".nrrd":
                    if helper.check_nrrd(path):  # is nrrd
                        paths, names = helper.slice_nrrd_file(path, converted_dir)
                        for path, name in zip(paths, names):
                            nrrd[name] = (path, None)
                elif mime == "application/gzip" or mime == "application/octet-stream":
                    if is_nifti_file(path):  # is nifti
                        paths, names = helper.slice_nifti_file(path, converted_dir)
                        for path, name in zip(paths, names):
                            nrrd[name] = (path, None)

        self._items = []
        for name, (path, group_tag) in nrrd.items():
            item = self.Item(item_path=path)
            if group_tag is not None:
                self._group_tag_names.append(group_tag.name)
                img_size = nrrd.read_header(path)["sizes"].tolist()[::-1]
                item.set_shape(img_size)
                item.ann_data = [group_tag]
            self._items.append(item)

        self._meta = meta
        if self.items_count == 0:
            remove_dir(converted_dir)

        return self.items_count > 0

    def _get_image_channels(self, file_path: str) -> List[np.ndarray]:
        file_ext = get_file_ext(file_path).lower()
        logger.debug(f"Working with file {file_path} with extension {file_ext}.")

        if file_ext == ".nrrd":
            logger.debug(f"Found nrrd file: {file_path}.")
            image, _ = nrrd.read(file_path)
        elif file_ext == ".dcm":
            pass
        return [image[:, :, i] for i in range(image.shape[2])]

    def to_supervisely(self, item, meta, *args) -> Union[Annotation, None]:
        ann = Annotation(item.shape)
        if item.ann_data is not None:
            ann = ann.add_tags(item.ann_data)
        return ann

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
    ) -> None:
        """Upload converted data to Supervisely"""

        if len(self._group_tag_names) > 0:
            if len(self._group_tag_names) > 1:
                # get the most common group tag name
                sorted_group_names = sorted(self._group_tag_names, key=lambda x: self._group_tag_names.count(x), reverse=True)
                logger.warn(
                    f"Multiple group tags found: {self._group_tag_names}. \n"
                    f"Will use: {sorted_group_names[0]}. \n"
                    "Some images will be hidden in the grouped view if they don't have the corresponding group tag."
                )
            meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)
            group_tag_name = renamed_tags.get(self._group_tag_names, self._group_tag_names)
            dataset = api.dataset.get_info_by_id(dataset_id)

            # ! FIXME: uncomment this line after adding set_medical_settings method to api.project
            # api.project.set_medical_settings(id=dataset.project_id, tag_name=group_tag_name)

            # backup:
            # api.project.images_grouping(id=dataset.project_id, enable=True, tag_name=group_tag_name)

            if log_progress:
                progress, progress_cb = self.get_progress(self.items_count, "Uploading images...")

            # existing_names = set([img.name for img in api.image.get_list(dataset_id)])

            for batch in batched(self._items, batch_size):
                paths = [item.path for item in batch]
                # ! FIXME: uncomment this line after adding upload_medical_images method to api.image
                # api.image.upload_medical_images(dataset_id, group_tag_name, paths, progress_cb=progress_cb)


                # backup:
                # anns = [self.to_supervisely(item, meta, renamed_classes, renamed_tags) for item in batch]
                # names = []
                # for item in batch:
                #     item.name = f"{get_file_name(item.path)}{get_file_ext(item.path).lower()}"
                #     name = generate_free_name(
                #         existing_names, item.name, with_ext=True, extend_used_names=True
                #     )
                #     names.append(name)

                # img_infos = api.image.upload_paths(dataset_id, names, paths)
                # img_ids = [img_info.id for img_info in img_infos]
                # api.annotation.upload_anns(img_ids, anns)

                if log_progress:
                    progress_cb(len(batch))

            if log_progress:
                if is_development():
                    progress.close()
            logger.info(f"Dataset ID:'{dataset_id}' has been successfully uploaded.")
        else:
            super().upload_dataset(api, dataset_id, batch_size, log_progress)