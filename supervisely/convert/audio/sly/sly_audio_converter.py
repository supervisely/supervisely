from typing import Dict, Optional

from supervisely._utils import generate_free_name, is_development
from supervisely.api.api import Api
from supervisely.audio_annotation.audio_annotation import AudioAnnotation
from supervisely.convert.audio.audio_converter import AudioConverter
from supervisely.convert.base_converter import AvailableAudioConverters
from supervisely.io.json import load_json_file
from supervisely.project.audio_project import AudioDataset, AudioProject
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger

DATASET_ITEMS = "items"
NESTED_DATASETS = "datasets"


class SLYAudioConverter(AudioConverter):
    """Imports a Supervisely audio project: ``meta.json`` plus datasets with
    ``audio/`` and ``ann/`` directories, as written by
    :func:`~supervisely.project.audio_project.download_audio_project`.

    The project's spectrogram settings in ``meta.json`` are carried over, see
    :meth:`AudioConverter._apply_spectrogram_settings`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Populated by read_sly_project when the project has more than one dataset:
        #   { ds_name: { "items": [Item, ...], "datasets": { child_name: {...} } } }
        self._project_structure = None

    def __str__(self) -> str:
        return AvailableAudioConverters.SLY

    @property
    def ann_ext(self) -> str:
        return ".json"

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta = None) -> bool:
        try:
            AudioAnnotation.load_json_file(ann_path, meta)
            return True
        except Exception as e:
            logger.warning(f"Failed to validate audio annotation {ann_path!r}: {repr(e)}")
            return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    def validate_format(self) -> bool:
        return self.read_sly_project(self._input_data)

    def read_sly_project(self, input_data: str) -> bool:
        try:
            project_fs = AudioProject.read_single(input_data)
            if project_fs.meta.project_type not in (None, ProjectType.AUDIO.value):
                return False
            self._meta = project_fs.meta
            self._items = []
            project = {}
            ds_cnt = 0

            for dataset_fs in project_fs.datasets:
                dataset_fs: AudioDataset
                ds_items = []
                for item_name in dataset_fs:
                    item_paths = dataset_fs.get_item_paths(item_name)
                    ann_path = item_paths.ann_path
                    if not self.validate_ann_file(ann_path, self._meta):
                        logger.warning(
                            f"Audio annotation for item {item_name!r} is invalid. "
                            "The item will be uploaded without labels."
                        )
                        ann_path = None
                    ds_items.append(self.Item(item_path=item_paths.audio_path, ann_data=ann_path))

                if ds_items:
                    # Dataset names on disk are path-style: "parent/child1" -> nested structure.
                    parts = dataset_fs.name.split("/")
                    curr_ds = project.setdefault(
                        parts[0], {DATASET_ITEMS: [], NESTED_DATASETS: {}}
                    )
                    for part in parts[1:]:
                        curr_ds = curr_ds[NESTED_DATASETS].setdefault(
                            part, {DATASET_ITEMS: [], NESTED_DATASETS: {}}
                        )
                    curr_ds[DATASET_ITEMS].extend(ds_items)
                    ds_cnt += 1
                    self._items.extend(ds_items)

            if self._items:
                if ds_cnt > 1:
                    self._project_structure = project
                return True
            return False
        except Exception as e:
            logger.info(f"Failed to read Supervisely audio project: {repr(e)}")
            return False

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
    ) -> None:
        if self._project_structure:
            self.upload_project(api, dataset_id, batch_size, log_progress)
        else:
            super().upload_dataset(api, dataset_id, batch_size, log_progress)

    def upload_project(
        self, api: Api, dataset_id: int, batch_size: int = 50, log_progress=True
    ) -> None:
        """Upload a multi-dataset audio project preserving the nested dataset hierarchy.

        The app pre-creates one dataset before calling this method. The first
        top-level dataset in the project is mapped onto that pre-created dataset
        (renamed in place); all remaining datasets and children are created fresh.
        """
        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id
        existing_datasets = {ds.name for ds in api.dataset.get_list(project_id, recursive=True)}

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading audio...")
        else:
            progress, progress_cb = None, None

        def _upload_project(
            project_structure: Dict,
            project_id: int,
            dataset_id: int,
            parent_id: Optional[int] = None,
            first_dataset: bool = False,
        ):
            for ds_name, value in project_structure.items():
                ds_name = generate_free_name(existing_datasets, ds_name, extend_used_names=True)
                if first_dataset:
                    first_dataset = False
                    api.dataset.update(dataset_id, ds_name)  # rename pre-created dataset
                else:
                    dataset_id = api.dataset.create(project_id, ds_name, parent_id=parent_id).id

                items = value.get(DATASET_ITEMS, [])
                nested = value.get(NESTED_DATASETS, {})
                logger.info(
                    f"Dataset: {ds_name}, items: {len(items)}, nested datasets: {len(nested)}"
                )
                if items:
                    super(SLYAudioConverter, self).upload_dataset(
                        api, dataset_id, batch_size, entities=items, progress_cb=progress_cb
                    )
                if nested:
                    _upload_project(nested, project_id, dataset_id, dataset_id)

        _upload_project(self._project_structure, project_id, dataset_id, first_dataset=True)

        if is_development() and progress is not None:
            progress.close()

    def to_supervisely(
        self,
        item: AudioConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> AudioAnnotation:
        if item.ann_data is None:
            return item.create_empty_annotation()
        try:
            ann = AudioAnnotation.load_json_file(item.ann_data)
        except Exception as e:
            logger.warning(f"Failed to read audio annotation: {repr(e)}")
            return item.create_empty_annotation()
        if renamed_tags:
            for tag in ann.tags + ann.recording_tags:
                tag.name = renamed_tags.get(tag.name, tag.name)
        return ann
