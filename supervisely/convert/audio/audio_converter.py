import os
from typing import Dict, List, Optional, Set, Tuple, Union

from supervisely._utils import batched, generate_free_name, is_development
from supervisely.api.api import Api
from supervisely.audio.audio_recording_tag import AudioRecordingTag
from supervisely.audio.audio_segment import AudioSegment
from supervisely.audio.spectrogram_settings import SpectrogramSettings
from supervisely.audio_annotation.audio_annotation import AudioAnnotation
from supervisely.convert.base_converter import BaseConverter
from supervisely.project.audio_project import ALLOWED_AUDIO_EXTENSIONS
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


class AudioConverter(BaseConverter):
    """Base converter for audio projects.

    Without a recognised annotation format it uploads every recording it finds,
    in any directory structure, without labels.
    """

    allowed_exts = list(ALLOWED_AUDIO_EXTENSIONS)
    modality = "audio"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Audio cannot be added by link: the files are always transferred. The
        # import manager has already downloaded remote input by the time a
        # converter runs, so the request is dropped here rather than letting
        # format detection skip every converter that "does not support links".
        self._links_requested = self._upload_as_links
        self._upload_as_links = False

    class Item(BaseConverter.BaseItem):
        """Audio item: recording path plus an optional annotation file."""

        def __init__(
            self,
            item_path: str,
            ann_data: Optional[str] = None,
            shape=None,
            custom_data: Optional[dict] = None,
        ):
            super().__init__(
                item_path=item_path,
                ann_data=ann_data,
                shape=shape,
                custom_data=custom_data,
            )
            self._type = "audio"

        def create_empty_annotation(self) -> AudioAnnotation:
            return AudioAnnotation()

    @property
    def format(self):
        return self._converter.format if self._converter is not None else self.__str__()

    def __str__(self) -> str:
        return self.modality

    @property
    def ann_ext(self):
        return None

    @property
    def key_file_ext(self):
        return None

    @staticmethod
    def validate_ann_file(ann_path, meta=None):
        return False

    @staticmethod
    def validate_key_file(key_file_path=None):
        return False

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
        entities: Optional[List[Item]] = None,
        progress_cb=None,
    ) -> None:
        """Upload recordings and their labels -- segments and whole-recording
        tags -- to a dataset.

        :param entities: Items to upload; defaults to all items collected by the
            converter. Pass a subset when uploading one dataset at a time.
        :param progress_cb: External progress callback. When provided, the
            method does **not** create its own progress bar.
        """
        if self._links_requested:
            logger.warning("Audio cannot be imported as links. The files will be uploaded.")

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)
        project_id = api.dataset.get_info_by_id(dataset_id).project_id
        self._apply_spectrogram_settings(api, project_id)
        tag_ids = self._get_tag_ids(api, project_id)

        existing_names = set(info.name for info in api.audio.get_list(dataset_id, recursive=False))

        _own_progress = None
        if progress_cb is None and log_progress:
            _own_progress, progress_cb = self.get_progress(
                len(entities or self._items), "Uploading audio..."
            )

        for batch in batched(entities or self._items, batch_size=batch_size):
            names, paths, anns = [], [], []
            for item in batch:
                item.name = generate_free_name(
                    existing_names, item.name, with_ext=True, extend_used_names=True
                )
                names.append(item.name)
                paths.append(item.path)
                anns.append(self.to_supervisely(item, meta, renamed_classes, renamed_tags))

            infos = api.audio.upload_paths(dataset_id, names, paths)
            for info, ann in zip(infos, anns):
                api.audio.add_tags(
                    project_id,
                    info.id,
                    segments=self._resolve_tag_ids(info.name, ann.tags, tag_ids),
                    recording_tags=self._resolve_tag_ids(info.name, ann.recording_tags, tag_ids),
                )

            if progress_cb is not None:
                progress_cb(len(batch))

        if _own_progress is not None and is_development():
            _own_progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")

    def to_supervisely(
        self,
        item: Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> AudioAnnotation:
        """Read the item's annotation, with tags renamed to match the project."""
        return item.create_empty_annotation()

    @staticmethod
    def _get_tag_ids(api: Api, project_id: int) -> Dict[str, int]:
        meta = ProjectMeta.from_json(api.project.get_meta(project_id))
        return {tag_meta.name: tag_meta.sly_id for tag_meta in meta.tag_metas}

    @staticmethod
    def _resolve_tag_ids(
        item_name: str,
        labels: List[Union[AudioSegment, AudioRecordingTag]],
        tag_ids: Dict[str, int],
    ) -> List[Union[AudioSegment, AudioRecordingTag]]:
        """Point each segment or recording tag at the destination project's
        tag by name.

        A label whose tag is not in the project cannot be written and is
        skipped with a warning, so one bad label does not fail the import.
        Ids carried over from the source project are discarded: they belong
        to another server.
        """
        resolved = []
        for label in labels:
            tag_id = tag_ids.get(label.name)
            if tag_id is None:
                where = (
                    f"Segment {label.start}-{label.end}"
                    if isinstance(label, AudioSegment)
                    else "Recording tag"
                )
                logger.warning(
                    f"{where} of {item_name!r} refers to tag {label.name!r}, "
                    "which is not in the project. The label is skipped."
                )
                continue
            label.tag_id = tag_id
            label.id = None
            label.entity_id = None
            resolved.append(label)
        return resolved

    def _apply_spectrogram_settings(self, api: Api, project_id: int) -> None:
        """Carry the source project's spectrogram settings to the destination.

        The settings decide what every label in the project means, so they are
        never changed under labels that already exist: they are written only
        when the destination project is not configured yet and holds no
        recordings. Otherwise the destination keeps its own, and a mismatch is
        logged.
        """
        imported = getattr(getattr(self._meta, "project_settings", None), "spectrogram", None)
        if imported is None:
            return
        try:
            imported = SpectrogramSettings.from_json(imported)
        except ValueError as e:
            logger.warning(f"Spectrogram settings in meta.json are invalid and are ignored: {e}")
            return

        current = api.project.get_settings(project_id).get("spectrogram")
        if current is None:
            if api.project.get_info_by_id(project_id).items_count:
                current = SpectrogramSettings()  # what the tool shows when unset
            else:
                api.audio.set_spectrogram_settings(project_id, imported)
                logger.info(f"Spectrogram settings applied to the project: {imported}")
                return
        else:
            current = SpectrogramSettings.from_json(current)

        if current.fingerprint != imported.fingerprint:
            logger.warning(
                "The project already has different spectrogram settings and keeps them: "
                f"project {current}, imported {imported}. Imported labels were drawn under "
                "the imported settings."
            )

    def _collect_items_if_format_not_detected(self) -> Tuple[List[Item], bool, Set[str]]:
        only_modality_items = True
        unsupported_exts = set()
        items = []

        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                if file in {"meta.json", "key_id_map.json"}:
                    continue
                ext = os.path.splitext(full_path)[1].lower()
                if ext in self.allowed_exts:
                    items.append(self.Item(full_path))
                else:
                    only_modality_items = False
                    if ext in self.unsupported_exts:
                        unsupported_exts.add(ext)

        return items, only_modality_items, unsupported_exts
