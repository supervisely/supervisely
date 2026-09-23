# coding: utf-8
"""API for audio projects.

Audio is **not** served by the image endpoints -- ``images.list`` answers
*"This API only supports images projects"*. Recordings come from
``entities.list`` and ``entities.info``, whose default projection omits both
``tags`` and ``meta``, so segment labels and the channel each one is about are
invisible unless the ``fields`` parameter asks for them explicitly.
:attr:`AudioApi.ENTITY_FIELDS` is that list.

The spectrogram is not on the labels. It is project configuration, read and
written with :meth:`AudioApi.get_spectrogram_settings` and
:meth:`AudioApi.set_spectrogram_settings`.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union

from tqdm import tqdm
from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.audio.audio_segment import AudioSegment
from supervisely.audio.spectrogram_settings import SpectrogramSettings
from supervisely.io.fs import ensure_base_path, get_file_hash


class AudioInfo(NamedTuple):
    """NamedTuple with information about an audio recording on the platform.

    The platform deliberately stores no sample rate, duration or channel count
    for audio -- ``file_meta`` is only ``{mime, size}``. Anything needing those
    decodes the file locally, see :func:`supervisely.audio.get_audio_info`.

    :Usage Example:

     .. code-block:: python

        AudioInfo(
            id=525,
            name="rain.wav",
            hash="Cz7LSAAtSlnpZH4sVBhRwrY=",
            link=None,
            dataset_id=98,
            project_id=31,
            workspace_id=12,
            created_at="2026-09-16T08:12:01.334Z",
            updated_at="2026-09-16T08:12:01.334Z",
            created_by_id=7,
            meta={},
            file_meta={"mime": "audio/wav", "size": 1536044},
            size=1536044,
            objects_count=0,
            path_original="/h5un/...wav",
            full_storage_url="http://localhost/h5un/...wav",
            tags=[],
        )
    """

    id: int
    name: str
    hash: str
    link: str
    dataset_id: int
    project_id: int
    workspace_id: int
    created_at: str
    updated_at: str
    created_by_id: int
    meta: dict
    file_meta: dict
    size: int
    objects_count: int
    path_original: str
    full_storage_url: str
    tags: list


class AudioApi(ModuleApiBase):
    """API for audio recordings and their segment labels.

    :param api: Parent :class:`Api` instance.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        api = sly.Api.from_env()
        recordings = api.audio.get_list(dataset_id)
        segments = api.audio.get_segments(recordings[0].id)
    """

    #: Field projection that makes ``entities.list`` and ``entities.info``
    #: return tags *and* their ``meta``. Without it there is no ``tags`` key at
    #: all, and with a partial list ``meta`` comes back missing.
    ENTITY_FIELDS = [
        "id",
        "name",
        "hash",
        "link",
        "datasetId",
        "projectId",
        "workspaceId",
        "createdAt",
        "updatedAt",
        "createdBy",
        "meta",
        "fileMeta",
        "size",
        "objectsCount",
        "pathOriginal",
        "fullStorageUrl",
        "tags",
    ]

    @staticmethod
    def info_sequence():
        """Get list of all :class:`AudioInfo` field names."""
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.HASH,
            ApiField.LINK,
            ApiField.DATASET_ID,
            ApiField.PROJECT_ID,
            ApiField.WORKSPACE_ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.CREATED_BY_ID,
            ApiField.META,
            ApiField.FILE_META,
            ApiField.SIZE,
            ApiField.OBJECTS_COUNT,
            ApiField.PATH_ORIGINAL,
            ApiField.FULL_STORAGE_URL,
            ApiField.TAGS,
        ]

    @staticmethod
    def info_tuple_name():
        """Get string name of :class:`AudioInfo` NamedTuple."""
        return "AudioInfo"

    def _convert_json_info(self, info: dict, skip_missing=True) -> AudioInfo:
        """Private method. Convert audio information from json to AudioInfo.

        ``entities.bulk.add`` answers with the *default* projection, which
        names the field ``title`` and leaves out ``tags``. Reading a recording
        back through :meth:`get_list` or :meth:`get_info_by_id` fills them in.
        """
        if info is not None and info.get(ApiField.NAME) is None and "title" in info:
            info = {**info, ApiField.NAME: info["title"]}
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        return AudioInfo(**res._asdict())

    # ------------------------------------------------------------------ read

    def get_list(
        self,
        dataset_id: int,
        filters: Optional[list[dict[str, str]]] = None,
        recursive: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> list[AudioInfo]:
        """List recordings in a dataset, with their segment labels attached.

        :param dataset_id: Dataset to list.
        :param filters: ``entities.list`` filters, as for images and videos.
        :param recursive: Include nested datasets.
        :param progress_cb: Function for tracking the listing progress.
        :return: list of :class:`AudioInfo`, each with ``tags`` filled in.
        """
        return self.get_list_all_pages(
            "entities.list",
            {
                ApiField.DATASET_ID: dataset_id,
                ApiField.FILTERS: filters or [],
                ApiField.FIELDS: self.ENTITY_FIELDS,
                "recursive": recursive,
            },
            progress_cb=progress_cb,
        )

    def get_info_by_id(self, id: int) -> Optional[AudioInfo]:
        """Fetch one recording with its tags.

        One request: ``entities.info`` answers for a single id and, given the
        ``fields`` projection, returns ``meta`` -- the field that carries each
        label's channel.

        :param id: Audio entity id.
        """
        return self._get_info_by_id(
            id, "entities.info", fields={ApiField.FIELDS: self.ENTITY_FIELDS}
        )

    def get_segments(self, entity_id: int) -> list[AudioSegment]:
        """Return the segment labels on a recording.

        Recording-level tags (no ``frameRange``) are skipped -- they are
        classifications of the whole file, not segments.

        :param entity_id: Audio entity id.
        """
        info = self.get_info_by_id(entity_id)
        if info is None:
            raise KeyError(f"audio entity {entity_id} not found")
        segments = []
        for tag in info.tags or []:
            if tag.get("frameRange") is None and tag.get("startFrame") is None:
                continue
            segments.append(AudioSegment.from_api_json(tag))
        return segments

    # ----------------------------------------------------------------- write

    def upload_path(self, dataset_id: int, name: str, path: str) -> AudioInfo:
        """Upload one local audio file into a dataset."""
        return self.upload_paths(dataset_id, [name], [path])[0]

    def upload_paths(
        self,
        dataset_id: int,
        names: list[str],
        paths: list[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> list[AudioInfo]:
        """Upload local audio files into a dataset.

        Files already on the server are recognised by hash and not sent again,
        exactly as for images and videos: re-running an interrupted upload
        transfers only what is missing.

        :param names: Names to give the recordings in the dataset.
        :param paths: Local paths, aligned with ``names``.
        :param progress_cb: Function for tracking the upload progress, counted
            in files.
        """
        if len(names) != len(paths):
            raise ValueError(f"got {len(names)} names for {len(paths)} paths")

        hashes = [get_file_hash(path) for path in paths]
        # Reuse the image uploader: the storage layer is shared and detects the
        # media type from the payload, so it also gives us hash de-duplication,
        # batching and retries.
        self._api.image._upload_data_bulk(
            lambda path: open(path, "rb"), zip(paths, hashes), progress_cb=progress_cb
        )
        return self.upload_hashes(dataset_id, names, hashes)

    def upload_hashes(
        self, dataset_id: int, names: list[str], hashes: list[str]
    ) -> list[AudioInfo]:
        """Add recordings that are already in storage to a dataset, by hash.

        The returned infos carry the endpoint's default projection: no ``tags``
        and no ``objects_count``. Re-read them with :meth:`get_info_by_id` when
        those are needed.
        """
        if len(names) != len(hashes):
            raise ValueError(f"got {len(names)} names for {len(hashes)} hashes")
        response = self._api.post(
            "entities.bulk.add",
            {
                ApiField.DATASET_ID: dataset_id,
                "entities": [{"name": n, "hash": h} for n, h in zip(names, hashes)],
            },
        ).json()
        return [self._convert_json_info(info) for info in response]

    def add_segments(
        self, project_id: int, entity_id: int, segments: list[AudioSegment]
    ) -> list[dict[str, Any]]:
        """Attach segment labels to a recording.

        A segment carries its inclusive sample range, its tag value and the
        channel it is about. The spectrogram it was drawn under is not stored
        per label: it is the project's, see
        :meth:`get_spectrogram_settings`.
        """
        if not segments:
            return []
        return self._api.post(
            "entities.tags.bulk.add",
            {
                ApiField.PROJECT_ID: project_id,
                "tags": [s._to_api_json(entity_id) for s in segments],
            },
        ).json()

    def add_segment(
        self, project_id: int, entity_id: int, segment: AudioSegment
    ) -> dict[str, Any]:
        """Attach a single segment label."""
        return self.add_segments(project_id, entity_id, [segment])[0]

    def remove_segment(self, tag_assignment_id: int) -> None:
        """Remove one segment label by its tag-assignment id."""
        self._api.post("image-tags.remove-from-image", {ApiField.ID: tag_assignment_id})

    # ---------------------------------------------------- project spectrogram

    def get_spectrogram_settings(self, project_id: int) -> SpectrogramSettings:
        """Read the spectrogram settings every recording in the project is
        analysed under.

        A project that has never been configured has no ``spectrogram`` key,
        and the platform defaults apply -- so this returns the defaults rather
        than ``None``, which is what the labeling tool shows in that case.

        :param project_id: Audio project id.

        :Usage example:

         .. code-block:: python

            settings = api.audio.get_spectrogram_settings(project_id)
            spec = sly.audio.render_spectrogram(samples, rate, settings)
        """
        stored = self._api.project.get_settings(project_id).get("spectrogram")
        return SpectrogramSettings.from_json(stored) if stored else SpectrogramSettings()

    def set_spectrogram_settings(
        self, project_id: int, settings: SpectrogramSettings
    ) -> None:
        """Configure the spectrogram for a whole audio project.

        This is a project-wide change: every annotator sees it, and existing
        labels are left untouched -- they were drawn on a picture that no
        longer matches. Set it once, at the start of the project.

        Requires permission to edit the project (``PROJECTS.UPDATE``), not just
        to label in it, and the project must be of type ``audio``.

        :param project_id: Audio project id.
        :param settings: Settings to apply.
        """
        if not isinstance(settings, SpectrogramSettings):
            raise TypeError(
                f"settings must be a SpectrogramSettings, got {type(settings).__name__}"
            )
        settings.validate()
        self._api.project.update_settings(
            project_id, {"spectrogram": settings.to_json()}, merge_with_current=True
        )

    # ------------------------------------------------------------- rendering

    def download_path(
        self, id: int, path: str, progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> None:
        """Download a recording's original file, unmodified.

        :param id: Audio entity id.
        :param path: Local path to write to.
        :param progress_cb: Function for tracking the download progress,
            counted in bytes.
        """
        response = self._api.post("images.download", {ApiField.ID: id}, stream=True)
        ensure_base_path(path)
        with open(path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)
                if progress_cb is not None:
                    progress_cb(len(chunk))

    def download_paths(
        self,
        ids: list[int],
        paths: list[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """Download several recordings to local paths."""
        if len(ids) != len(paths):
            raise ValueError(f"got {len(ids)} ids for {len(paths)} paths")
        for id_, path in zip(ids, paths):
            self.download_path(id_, path, progress_cb=progress_cb)
