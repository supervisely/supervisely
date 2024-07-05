import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

import magic
import nrrd
import numpy as np

from supervisely import Annotation, batched, generate_free_name, is_development, ProjectMeta, logger, TagMeta, Tag, TagValueType
from supervisely.api.api import Api, ApiContext
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.medical2d import medical2d_helper as helper
from supervisely.io.fs import remove_dir, get_file_ext, mkdir, get_file_name
from supervisely.project.project_settings import LabelingInterface
from supervisely.volume.volume import is_nifti_file


# @TODO: add group tags?
class Medical2DImageConverter(ImageConverter):

    def __init__(
            self,
            input_data: str,
            labeling_interface: Optional[Union[LabelingInterface, str]],
            upload_as_links: bool,
            remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        self._filtered = None
        self._group_tag_names = defaultdict(int)

    def __str__(self):
        return AvailableImageConverters.MEDICAL2D

    def validate_labeling_interface(self) -> bool:
        """Only medical labeling interface can be used for medical images."""
        return self._labeling_interface in [
            LabelingInterface.DEFAULT,
            LabelingInterface.MEDICAL_IMAGING_SINGLE,
            LabelingInterface.IMAGES_WITH_16_COLOR,
        ]

    def validate_format(self) -> bool:
        logger.debug(f"Validating format: {self.__str__()}")

        converted_dir_name = "converted"
        converted_dir = os.path.join(self._input_data, converted_dir_name)
        mkdir(converted_dir, remove_content_if_exists=True)
        meta = ProjectMeta()

        nrrds_dict = {}
        for root, _, files in os.walk(self._input_data):
            if converted_dir_name in root:
                continue
            for file in files:
                path = os.path.join(root, file)
                ext = get_file_ext(path).lower()
                mime = magic.from_file(path, mime=True)
                if mime == "application/dicom":
                    if helper.is_dicom_file(path):  # is dicom
                        paths, names, group_tags, dcm_metas = helper.convert_dcm_to_nrrd(path, converted_dir)
                        for group_tag in group_tags:
                            if meta.get_tag_meta(group_tag["name"]) is None:
                                tag_meta = TagMeta(group_tag["name"], TagValueType.ANY_STRING)
                                meta = meta.add_tag_meta(tag_meta)
                        for path, name in zip(paths, names):
                            nrrds_dict[name] = (path, group_tags, dcm_metas)
                elif ext == ".nrrd":
                    if helper.check_nrrd(path):  # is nrrd
                        paths, names = helper.slice_nrrd_file(path, converted_dir)
                        for path, name in zip(paths, names):
                            nrrds_dict[name] = (path, None, None)
                elif mime == "application/gzip" or mime == "application/octet-stream":
                    if is_nifti_file(path):  # is nifti
                        paths, names = helper.slice_nifti_file(path, converted_dir)
                        for path, name in zip(paths, names):
                            nrrds_dict[name] = (path, None, None)

        self._items = []
        for name, (path, group_tags, dcm_metas) in nrrds_dict.items():
            item = self.Item(item_path=path)
            img_size = nrrd.read_header(path)["sizes"].tolist()[::-1] # pylint: disable=no-member
            item.set_shape(img_size)
            if group_tags is not None:
                for group_tag in group_tags:
                    self._group_tag_names[group_tag["name"]] += 1
                item.ann_data = group_tags
            item.set_meta_data(dcm_metas if dcm_metas is not None else {})
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

    def to_supervisely(
        self,
        item,
        meta: ProjectMeta,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Annotation:
        ann = Annotation(item.shape)
        if item.ann_data is not None:
            tags = []
            for tag in item.ann_data:
                tag_name = renamed_tags.get(tag["name"], tag["name"])
                tag_meta = meta.get_tag_meta(tag_name)
                if tag_meta is not None:
                    group_tag = Tag(tag_meta, str(tag["value"]))
                    tags.append(group_tag)
            ann = ann.add_tags(tags)
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
            logger.debug("Group tags detected")
            group_tag_name = next(iter(self._group_tag_names))
            if len(self._group_tag_names) > 1:
                group_tag_name = max(self._group_tag_names, key=self._group_tag_names.get)
                logger.warn(
                    f"Multiple metadata fields found: {', '.join(self._group_tag_names.keys())}..."
                    "Some images will be hidden in the grouped view if they don't have the corresponding group tag."
                )
            meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)
            group_tag_name = renamed_tags.get(group_tag_name, group_tag_name)
            logger.info(f"Will use [{group_tag_name}] as group tag.")

            dataset = api.dataset.get_info_by_id(dataset_id)
            existing_names = set([img.name for img in api.image.get_list(dataset_id)])

            api.project.images_grouping(id=dataset.project_id, enable=True, tag_name=group_tag_name)


            if log_progress:
                progress, progress_cb = self.get_progress(self.items_count, "Uploading images...")

            for batch in batched(self._items, batch_size):
                paths = [item.path for item in batch]
                anns = [self.to_supervisely(item, meta, renamed_classes, renamed_tags) for item in batch]
                img_metas = [item.meta or {} for item in batch]
                names = []
                for item in batch:
                    item.name = f"{get_file_name(item.path)}{get_file_ext(item.path).lower()}"
                    name = generate_free_name(
                        existing_names, item.name, with_ext=True, extend_used_names=True
                    )
                    names.append(name)

                with ApiContext(
                    api=api, project_id=dataset.project_id, dataset_id=dataset_id, project_meta=meta
                ):
                    img_infos = api.image.upload_paths(dataset_id, names, paths, metas=img_metas)
                    img_ids = [img_info.id for img_info in img_infos]
                    api.annotation.upload_anns(img_ids, anns)

                if log_progress:
                    progress_cb(len(batch))

            if log_progress:
                if is_development():
                    progress.close()
            logger.info(f"Dataset ID:'{dataset_id}' has been successfully uploaded.")
        else:
            super().upload_dataset(api, dataset_id, batch_size, log_progress)