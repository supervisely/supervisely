"""Framework-agnostic helpers for the video-aware live training endpoints.

These helpers are intentionally subclass-independent so the SDK can serve
``/predict-video``, ``/highlight_key_frames``, and ``/inference_video_id``
without dragging detection-specific code into
the base class.
"""

from typing import List, Optional
import uuid

import numpy as np
from tqdm import tqdm

import supervisely as sly
from supervisely.api.module_api import ApiField
from supervisely.api.video.video_tag_api import VideoObjectTagApi


def uniform_sample_indices(
    n_frames: int,
    min_samples: int = 30,
    max_samples: int = 150,
    step: int = 10,
) -> List[int]:
    indices = np.arange(0, n_frames, step, dtype=int)
    n_selected = len(indices)

    if n_selected < min_samples:
        target = min(min_samples, n_frames)
        indices = np.linspace(0, n_frames - 1, num=target, dtype=int)
    elif n_selected > max_samples:
        indices = np.linspace(0, n_frames - 1, num=max_samples, dtype=int)

    indices = np.unique(indices)
    return indices.tolist()


def get_uniform_frame_indices(video_id: int, api: sly.Api) -> List[int]:
    video_info = api.video.get_info_by_id(video_id)
    return uniform_sample_indices(video_info.frames_count)


def load_uniform_video_frames(
    video_id: int,
    api: sly.Api,
    uniform_indices: Optional[List[int]] = None,
) -> List[np.ndarray]:
    if not uniform_indices:
        uniform_indices = get_uniform_frame_indices(video_id, api)
    with tqdm(message="Downloading video frames...", total=len(uniform_indices)) as pbar:
        frames = api.video.frame.download_nps(video_id, uniform_indices, progress_cb=pbar)
    return frames


def refresh_meta(
    api: sly.Api,
    project_id: int,
    project_meta: sly.ProjectMeta,
    new_tag_meta: sly.TagMeta,
):
    """Ensure ``new_tag_meta`` is present in the project meta and return the
    canonical (server-side) version plus the refreshed project meta.
    """
    if not project_meta.tag_metas.has_key(new_tag_meta.name):
        new_tags_collection = project_meta.tag_metas.add(new_tag_meta)
        project_meta = sly.ProjectMeta(
            tag_metas=new_tags_collection, obj_classes=project_meta.obj_classes
        )
        api.project.update_meta(project_id, project_meta)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        new_tag_meta = project_meta.tag_metas.get(new_tag_meta.name)
    else:
        tag_values = new_tag_meta.possible_values
        new_tag_meta = project_meta.tag_metas.get(new_tag_meta.name)
        if tag_values and sorted(new_tag_meta.possible_values) != sorted(tag_values):
            sly.logger.warning(
                f"Tag [{new_tag_meta.name}] already exists, but with another values: "
                f"{new_tag_meta.possible_values}"
            )
    return new_tag_meta, project_meta


def filter_objects_by_confidence(
    objects_json: list,
    project_meta: sly.ProjectMeta,
    threshold: float,
) -> List[sly.Label]:
    """Drop predictions whose confidence is below ``threshold`` and return the
    surviving figures as ``sly.Label`` instances.
    """
    return [
        sly.Label.from_json(obj, project_meta)
        for obj in objects_json
        if obj["meta"]["confidence"] >= threshold
    ]


def create_video_object(api: sly.Api, video_info, obj_class: sly.ObjClass) -> int:
    """Create a single new ``VideoObject`` of the given class on the video and
    return its server-side id.
    """
    vid_obj = sly.VideoObject(obj_class)
    obj_ids = api.video.object._append_bulk(
        tag_api=VideoObjectTagApi(api),
        entity_id=video_info.id,
        project_id=video_info.project_id,
        dataset_id=video_info.dataset_id,
        objects=sly.VideoObjectCollection([vid_obj]),
        key_id_map=sly.KeyIdMap(),
    )
    return obj_ids[0]


def label_to_video_figure_json(label: sly.Label, object_id: int, frame_index: int) -> dict:
    geometry = label.geometry
    return {
        ApiField.OBJECT_ID: object_id,
        ApiField.GEOMETRY_TYPE: geometry.geometry_name(),
        ApiField.GEOMETRY: geometry.to_json(),
        ApiField.META: {ApiField.FRAME: frame_index},
        ApiField.NN_CREATED: True,
        ApiField.NN_UPDATED: True,
    }


def upload_video_figures(
    api: sly.Api,
    video_id: int,
    figures_json: List[dict],
    toolbox_session_id: Optional[str] = None,
    batch_size: int = 100,
) -> List[int]:
    """Bulk-POST figures to ``figures.bulk.add`` and return the created ids."""
    if not figures_json:
        return []

    figures_keys = [uuid.uuid4() for _ in figures_json]
    key_id_map = sly.KeyIdMap()
    figure_ids: List[int] = []

    if toolbox_session_id is not None:
        api.headers["x-toolbox-session-id"] = toolbox_session_id

    payload_extra = {}
    if toolbox_session_id is not None:
        payload_extra["toolboxSessionId"] = toolbox_session_id

    for batch_keys, batch_jsons in zip(
        sly.batched(figures_keys, batch_size=batch_size),
        sly.batched(figures_json, batch_size=batch_size),
    ):
        resp = api.post(
            "figures.bulk.add",
            {
                ApiField.ENTITY_ID: video_id,
                ApiField.FIGURES: batch_jsons,
                **payload_extra,
            },
        )
        for key, resp_obj in zip(batch_keys, resp.json()):
            figure_id = resp_obj[ApiField.ID]
            key_id_map.add_figure(key, figure_id)
            figure_ids.append(figure_id)

    return figure_ids


def remove_video_figures(
    api: sly.Api,
    video_id: int,
    figure_ids: List[int],
    toolbox_session_id: Optional[str] = None,
    batch_size: int = 100,
) -> None:
    """Bulk-DELETE figures via ``figures.bulk.remove``. Same endpoint the
    SDK's FigureApi.remove_batch hits, but we send it directly so we can
    attach the toolbox session id (the labeling UI needs it for undo/redo).
    """
    if not figure_ids:
        return

    if toolbox_session_id is not None:
        api.headers["x-toolbox-session-id"] = toolbox_session_id

    payload_extra = {}
    if toolbox_session_id is not None:
        payload_extra["toolboxSessionId"] = toolbox_session_id

    for batch in sly.batched(list(figure_ids), batch_size=batch_size):
        api.post(
            "figures.bulk.remove",
            {ApiField.FIGURE_IDS: list(batch), **payload_extra},
        )
