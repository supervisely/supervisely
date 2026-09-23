# coding: utf-8
"""Local Supervisely project for audio recordings.

Layout on disk mirrors the other modalities::

    project/
        meta.json
        ds0/
            audio/rain.wav
            ann/rain.wav.json
            audio_info/rain.wav.json     # only with save_audio_info=True

``ann/<name>.json`` holds an :class:`~supervisely.audio_annotation.audio_annotation.AudioAnnotation`:
the shape of the recording plus its segment labels. Tags are stored by **name**,
so a downloaded project is readable without the server that issued the ids.

The project's spectrogram settings travel in ``meta.json`` under
``projectSettings.spectrogram``, because that is where the platform keeps them:
they are project configuration applied to every recording, not part of any one
label. Download writes them out and upload puts them back, so a round trip
preserves the analysis the labels were drawn under.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, NamedTuple, Optional, Union

from tqdm import tqdm

from supervisely._utils import batched
from supervisely.api.api import Api
from supervisely.api.audio_api import AudioInfo
from supervisely.audio.audio_io import get_audio_info
from supervisely.audio.audio_segment import AudioSegment
from supervisely.audio_annotation.audio_annotation import AudioAnnotation
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.io.fs import get_file_ext, touch
from supervisely.io.json import dump_json_file
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project import read_single_project as read_project_wrapper
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly

#: Extensions the platform accepts for audio recordings.
ALLOWED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

LOG_BATCH_SIZE = 20


class AudioItemPaths(NamedTuple):
    """Paths to an audio item and its annotation file."""

    audio_path: str
    ann_path: str


class AudioDataset(Dataset):
    """A dataset directory for audio recordings inside a local audio project."""

    item_dir_name = "audio"
    ann_dir_name = "ann"
    item_info_dir_name = "audio_info"
    seg_dir_name = None

    annotation_class = AudioAnnotation
    item_info_class = AudioInfo

    @property
    def audio_dir(self) -> str:
        """Path to the directory with audio recordings."""
        return self.item_dir

    @property
    def audio_info_dir(self) -> str:
        """Path to the directory with recording info files."""
        return self.item_info_dir

    @property
    def img_dir(self) -> str:
        """Not supported for audio datasets."""
        raise NotImplementedError(
            f"Property 'img_dir' is not supported for {type(self).__name__} object."
        )

    @property
    def img_info_dir(self):
        """Not supported for audio datasets."""
        raise NotImplementedError(
            f"Property 'img_info_dir' is not supported for {type(self).__name__} object."
        )

    @property
    def seg_dir(self):
        """Not supported for audio datasets."""
        raise NotImplementedError(
            f"Property 'seg_dir' is not supported for {type(self).__name__} object."
        )

    @classmethod
    def _has_valid_ext(cls, path: str) -> bool:
        """Check whether the given path has an allowed audio file extension."""
        return get_file_ext(path).lower() in ALLOWED_AUDIO_EXTENSIONS

    def get_ann(
        self, item_name: str, project_meta: ProjectMeta, key_id_map=None
    ) -> AudioAnnotation:
        """Read the annotation of an item from json."""
        return self.annotation_class.load_json_file(
            self.get_ann_path(item_name), project_meta, key_id_map
        )

    def set_ann(self, item_name: str, ann: AudioAnnotation, key_id_map=None) -> None:
        """Write the annotation of an item to json."""
        if type(ann) is not self.annotation_class:
            raise TypeError(
                f"Type of 'ann' should be {self.annotation_class.__name__}, "
                f"not a {type(ann).__name__}"
            )
        dump_json_file(ann.to_json(), self.get_ann_path(item_name), indent=4)

    def _get_empty_annotaion(self, item_name: str) -> AudioAnnotation:
        """Empty annotation, with the recording's shape read from its header."""
        item_path = self.get_item_path(item_name)
        try:
            info = get_audio_info(item_path)
        except Exception:
            # An item added as a placeholder (download_audios=False) has no
            # decodable header; the shape stays unknown rather than wrong.
            return AudioAnnotation()
        return AudioAnnotation(
            sample_count=info.sample_count,
            sample_rate=info.sample_rate,
            channels=info.channels,
        )

    def add_item_np(self, item_name, img, ann=None, img_info=None):
        """Not available for AudioDataset."""
        raise NotImplementedError(
            f"Method 'add_item_np()' is not supported for {type(self).__name__} object."
        )

    def add_item_raw_bytes(self, item_name, item_raw_bytes, ann=None, img_info=None):
        """Not available for AudioDataset."""
        raise NotImplementedError(
            f"Method 'add_item_raw_bytes()' is not supported for {type(self).__name__} object."
        )

    def _add_img_np(self, item_name, img):
        """Not available for AudioDataset."""
        raise NotImplementedError(
            f"Method '_add_img_np()' is not supported for {type(self).__name__} object."
        )

    def _add_item_raw_bytes(self, item_name, item_raw_bytes):
        """Not available for AudioDataset."""
        raise NotImplementedError(
            f"Method '_add_item_raw_bytes()' is not supported for {type(self).__name__} object."
        )

    def _validate_added_item_or_die(self, item_path: str) -> None:
        """Reject anything that is not an audio file by extension."""
        if not self._has_valid_ext(item_path):
            os.remove(item_path)
            raise RuntimeError(
                f"Unsupported audio extension {get_file_ext(item_path)!r}, "
                f"expected one of {ALLOWED_AUDIO_EXTENSIONS}"
            )

    def get_item_paths(self, item_name: str) -> AudioItemPaths:
        """Paths to the item and its annotation."""
        return AudioItemPaths(
            audio_path=self.get_item_path(item_name), ann_path=self.get_ann_path(item_name)
        )

    def get_item_info(self, item_name: str) -> AudioInfo:
        """Read the stored :class:`AudioInfo` of an item."""
        info_path = self.get_item_info_path(item_name)
        from supervisely.io.json import load_json_file

        return AudioInfo(**load_json_file(info_path))


class AudioProject(Project):
    """A local Supervisely project for audio recordings."""

    dataset_class = AudioDataset

    class DatasetDict(KeyIndexedCollection):
        """Key-indexed collection of :class:`AudioDataset` datasets."""

        item_type = AudioDataset

    @property
    def type(self) -> str:
        """Project type: ``audio``."""
        return ProjectType.AUDIO.value

    @classmethod
    def read_single(cls, dir: str) -> "AudioProject":
        """Read a single audio project from a directory."""
        return read_project_wrapper(dir, cls)

    @staticmethod
    def download(
        api: Api,
        project_id: int,
        dest_dir: str,
        dataset_ids: Optional[List[int]] = None,
        download_audios: bool = True,
        save_audio_info: bool = False,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """Download an audio project from Supervisely into a local directory."""
        download_audio_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            download_audios=download_audios,
            save_audio_info=save_audio_info,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    @staticmethod
    def upload(
        dir: str,
        api: Api,
        workspace_id: int,
        project_name: Optional[str] = None,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ):
        """Upload a local audio project to Supervisely."""
        return upload_audio_project(
            dir=dir,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )


def _tag_names_by_id(api: Api, project_id: int) -> Dict[int, str]:
    """Map server tag ids to tag names, so downloaded labels carry names."""
    meta_json = api.project.get_meta(project_id)
    return {tag["id"]: tag["name"] for tag in meta_json.get("tags", []) if "id" in tag}


def download_audio_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    download_audios: bool = True,
    save_audio_info: bool = False,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> None:
    """Download an audio project from Supervisely into a local directory.

    :param api: Supervisely API object.
    :param project_id: Project ID in Supervisely.
    :param dest_dir: Directory to download the project into.
    :param dataset_ids: Datasets to download; all of them when omitted.
    :param download_audios: Download the recordings themselves. With ``False``
        only empty placeholder files are created, and an annotation then has no
        sample rate to convert its ranges to seconds.
    :param save_audio_info: Save an ``AudioInfo`` json next to each recording.
    :param log_progress: Log the download progress.
    :param progress_cb: Function for tracking the download progress.
    """
    # with_settings=True is what carries `projectSettings.spectrogram` into
    # meta.json; without it the analysis the labels were drawn under is lost.
    meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))
    tag_names = _tag_names_by_id(api, project_id)

    project_fs = AudioProject(dest_dir, OpenMode.CREATE)
    project_fs.set_meta(meta)

    if progress_cb is not None:
        log_progress = False

    dataset_ids = set(dataset_ids) if (dataset_ids is not None) else None
    for parents, dataset in api.dataset.tree(project_id):
        if dataset_ids is not None and dataset.id not in dataset_ids:
            continue

        dataset_path = Dataset._get_dataset_path(dataset.name, parents)
        dataset_fs = project_fs.create_dataset(dataset.name, dataset_path)
        recordings = api.audio.get_list(dataset.id, recursive=False)

        ds_progress = progress_cb
        if log_progress:
            ds_progress = tqdm_sly(
                desc="Downloading audio from {!r}".format(dataset.name),
                total=len(recordings),
            )

        for batch in batched(recordings, batch_size=LOG_BATCH_SIZE):
            for info in batch:
                audio_path = dataset_fs.generate_item_path(info.name)
                if download_audios:
                    api.audio.download_path(info.id, audio_path)
                else:
                    touch(audio_path)

                ann = _build_annotation(info, audio_path, tag_names, download_audios)
                item_info = info._asdict() if save_audio_info else None
                dataset_fs.add_item_file(
                    info.name,
                    audio_path,
                    ann=ann,
                    _validate_item=False,
                    _use_hardlink=True,
                    item_info=item_info,
                )
                if progress_cb is not None:
                    progress_cb(1)

            if log_progress:
                ds_progress(len(batch))


def _build_annotation(
    info: AudioInfo, audio_path: str, tag_names: Dict[int, str], decoded: bool
) -> AudioAnnotation:
    """Turn a recording's server tags into a local annotation."""
    segments = []
    for tag in info.tags or []:
        if tag.get("frameRange") is None and tag.get("startFrame") is None:
            continue  # a recording-level tag, not a segment
        segment = AudioSegment.from_api_json(tag)
        segment.name = tag_names.get(segment.tag_id)
        segments.append(segment)

    shape = {}
    if decoded:
        try:
            file_info = get_audio_info(audio_path)
            shape = {
                "sample_count": file_info.sample_count,
                "sample_rate": file_info.sample_rate,
                "channels": file_info.channels,
            }
        except Exception as e:
            # The platform stores no audio metadata, so an undecodable file
            # leaves the shape unknown rather than guessed.
            logger.warning(f"Failed to read the header of {audio_path!r}: {e}")
    return AudioAnnotation(tags=segments, **shape)


def upload_audio_project(
    dir: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
):
    """Upload a local audio project to Supervisely.

    :param dir: Local project directory.
    :param api: Supervisely API object.
    :param workspace_id: Destination workspace.
    :param project_name: Name for the created project; the directory name when
        omitted.
    :param log_progress: Log the upload progress.
    :param progress_cb: Function for tracking the upload progress.
    :return: ``(project_id, project_name)`` of the created project.
    """
    project_fs = AudioProject.read_single(dir)
    if project_name is None:
        project_name = project_fs.name

    project = api.project.create(
        workspace_id, project_name, type=ProjectType.AUDIO, change_name_if_conflict=True
    )
    api.project.update_meta(project.id, project_fs.meta.to_json())

    # update_meta does not carry project settings, so the spectrogram has to be
    # written separately. It needs permission to edit the project, which the
    # uploader has: it just created it.
    spectrogram = getattr(project_fs.meta.project_settings, "spectrogram", None)
    if spectrogram is not None:
        api.project.update_settings(
            project.id, {"spectrogram": spectrogram}, merge_with_current=True
        )

    tag_ids = {name: id for id, name in _tag_names_by_id(api, project.id).items()}

    for dataset_fs in project_fs.datasets:
        dataset = api.dataset.create(project.id, dataset_fs.name, change_name_if_conflict=True)

        names = list(dataset_fs.get_items_names())
        paths = [dataset_fs.get_item_path(name) for name in names]

        ds_progress = progress_cb
        if log_progress and progress_cb is None:
            ds_progress = tqdm_sly(
                desc="Uploading audio to {!r}".format(dataset.name), total=len(names)
            )

        uploaded = api.audio.upload_paths(dataset.id, names, paths, progress_cb=ds_progress)

        for name, info in zip(names, uploaded):
            ann = dataset_fs.get_ann(name, project_fs.meta)
            segments = []
            for segment in ann.tags:
                if segment.name is None:
                    raise RuntimeError(
                        f"Segment of {name!r} has no tag name, so it cannot be uploaded"
                    )
                if segment.name not in tag_ids:
                    raise RuntimeError(f"Tag {segment.name!r} is missing from the project meta")
                segment.tag_id = tag_ids[segment.name]
                segments.append(segment)
            if segments:
                api.audio.add_segments(project.id, info.id, segments)

    return project.id, project.name
