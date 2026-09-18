# coding: utf-8
"""API for audio projects.

Audio is **not** served by the image endpoints -- ``images.list`` answers
*"This API only supports images projects"*. Recordings come from
``entities.list``, and its default projection omits both ``tags`` and ``meta``,
so segment labels and their spectrogram settings are invisible unless the
``fields`` parameter asks for them explicitly. :attr:`AudioApi.ENTITY_FIELDS`
is that list.
"""

from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from typing import Any, Dict, List, Optional

from requests_toolbelt import MultipartEncoder

from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.audio.audio_segment import AudioSegment
from supervisely.io.fs import ensure_base_path, get_file_name_with_ext


class AudioApi(ModuleApiBase):
    """API for audio recordings and their segment labels.

    :param api: Parent :class:`Api` instance.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        api = sly.Api.from_env()
        recordings = api.audio.get_list(dataset_id)
        segments = api.audio.get_segments(recordings[0]["id"], dataset_id)
    """

    #: Field projection that makes ``entities.list`` return tags *and* their
    #: ``meta``. Copied from the labeling tool; without it there is no ``tags``
    #: key at all, and with a partial list ``meta`` comes back missing.
    ENTITY_FIELDS = [
        "id",
        "title",
        "description",
        "parentId",
        "projectId",
        "datasetId",
        "createdAt",
        "updatedAt",
        "createdBy",
        "meta",
        "pathOriginal",
        "hash",
        "fileMeta",
        "fullStorageUrl",
        "link",
        "workspaceId",
        "objectsCount",
        "tags",
    ]

    def __init__(self, api):
        super().__init__(api)
        self._api = api

    # ------------------------------------------------------------------ read

    def get_list(self, dataset_id: int, recursive: bool = True) -> List[Dict[str, Any]]:
        """List recordings in a dataset, with their segment labels attached.

        :param dataset_id: Dataset to list.
        :param recursive: Include nested datasets.
        :return: Raw entity dicts, each with a ``tags`` key.
        """
        entities: List[Dict[str, Any]] = []
        page = 1
        while True:
            # `fields` has to travel as a query parameter: entities.list only
            # returns `tags` (and their `meta`) when asked for them explicitly.
            query = urlencode(
                {"fields": json.dumps(self.ENTITY_FIELDS), "sort": "id", "page": page}
            )
            response = self._api.post(
                f"entities.list?{query}",
                {"datasetId": dataset_id, "recursive": recursive},
            ).json()
            entities.extend(response.get("entities", []))
            if page >= response.get("pagesCount", 1):
                break
            page += 1
        return entities

    def get_info_by_id(self, id: int, dataset_id: int) -> Optional[Dict[str, Any]]:
        """Fetch one recording with its tags.

        A ``dataset_id`` is required because the only projection that returns
        ``meta`` is the list endpoint. ``images.info`` would answer for a single
        id but silently drops ``meta``, which is exactly the field that carries
        the spectrogram settings.
        """
        for entity in self.get_list(dataset_id):
            if entity["id"] == id:
                return entity
        return None

    def get_segments(self, entity_id: int, dataset_id: int) -> List[AudioSegment]:
        """Return the segment labels on a recording.

        Recording-level tags (no ``frameRange``) are skipped -- they are
        classifications of the whole file, not segments.
        """
        entity = self.get_info_by_id(entity_id, dataset_id)
        if entity is None:
            raise KeyError(f"audio entity {entity_id} not found in dataset {dataset_id}")
        segments = []
        for tag in entity.get("tags") or []:
            if tag.get("frameRange") is None and tag.get("startFrame") is None:
                continue
            segments.append(AudioSegment.from_api_json(tag))
        return segments

    # ----------------------------------------------------------------- write

    def upload_path(self, dataset_id: int, name: str, path: str) -> Dict[str, Any]:
        """Upload one local audio file into a dataset."""
        return self.upload_paths(dataset_id, [name], [path])[0]

    def upload_paths(
        self, dataset_id: int, names: List[str], paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Upload local audio files into a dataset.

        :param names: Names to give the recordings in the dataset.
        :param paths: Local paths, aligned with ``names``.
        """
        if len(names) != len(paths):
            raise ValueError(f"got {len(names)} names for {len(paths)} paths")

        hashes = []
        for path in paths:
            with open(path, "rb") as fh:
                encoder = MultipartEncoder(
                    fields={
                        get_file_name_with_ext(path): (
                            os.path.basename(path),
                            fh,
                            "audio/*",
                        )
                    }
                )
                response = self._api.post("images.bulk.upload", encoder)
            payload = response.json()
            errors = payload[0].get("errors")
            if errors:
                raise RuntimeError(f"upload of {path!r} failed: {errors}")
            hashes.append(payload[0]["hash"])

        return self._api.post(
            "entities.bulk.add",
            {
                ApiField.DATASET_ID: dataset_id,
                "entities": [{"name": n, "hash": h} for n, h in zip(names, hashes)],
            },
        ).json()

    def add_segments(
        self, project_id: int, entity_id: int, segments: List[AudioSegment]
    ) -> List[Dict[str, Any]]:
        """Attach segment labels to a recording.

        Each segment's spectrogram settings, when present, are written into the
        tag assignment's ``meta`` so the label records the view it was made
        under. Settings are validated client-side first: the API currently
        accepts values the labeling tool then refuses to open.
        """
        if not segments:
            return []
        for segment in segments:
            if segment.settings is not None:
                segment.settings.validate()
        return self._api.post(
            "entities.tags.bulk.add",
            {
                ApiField.PROJECT_ID: project_id,
                "tags": [s._to_api_json(entity_id) for s in segments],
            },
        ).json()

    def add_segment(
        self, project_id: int, entity_id: int, segment: AudioSegment
    ) -> Dict[str, Any]:
        """Attach a single segment label."""
        return self.add_segments(project_id, entity_id, [segment])[0]

    def remove_segment(self, tag_assignment_id: int) -> None:
        """Remove one segment label by its tag-assignment id."""
        self._api.post("image-tags.remove-from-image", {ApiField.ID: tag_assignment_id})

    # ------------------------------------------------------------- rendering

    def download_path(self, id: int, path: str) -> None:
        """Download a recording's original file, unmodified.

        The platform deliberately stores no sample rate, duration or channel
        count for audio, so anything needing those decodes the file locally --
        see :func:`supervisely.audio.get_audio_info`.

        :param id: Audio entity id.
        :param path: Local path to write to.
        """
        response = self._api.post("images.download", {ApiField.ID: id}, stream=True)
        ensure_base_path(path)
        with open(path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

    def download_paths(self, ids: List[int], paths: List[str]) -> None:
        """Download several recordings to local paths."""
        if len(ids) != len(paths):
            raise ValueError(f"got {len(ids)} ids for {len(paths)} paths")
        for id_, path in zip(ids, paths):
            self.download_path(id_, path)
