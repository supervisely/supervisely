import asyncio
import concurrent.futures
import os
import threading
import urllib.parse
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from supervisely import logger
from supervisely._utils import batched_iter, run_coroutine
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_collection import TagCollection
from supervisely.annotation.tag_meta import TagApplicableTo, TagMeta, TagValueType
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.image_api import ImageInfo
from supervisely.api.module_api import ApiField
from supervisely.api.project_api import ProjectInfo
from supervisely.api.video.video_api import VideoInfo
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import tqdm_sly
from supervisely.video.video import VideoFrameReader
from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation

VIDEO_OBJECT_TAG_META = TagMeta(
    "object_id",
    value_type=TagValueType.ANY_NUMBER,
    applicable_to=TagApplicableTo.OBJECTS_ONLY,
)
AUTO_TRACKED_TAG_META = TagMeta(
    "auto-tracked", TagValueType.NONE, applicable_to=TagApplicableTo.OBJECTS_ONLY
)


class ApiContext:
    """Small per-run cache of API-fetched project/dataset/video metadata to avoid repeated requests."""

    def __init__(self):
        self.project_info: Dict[int, ProjectInfo] = {}
        self.project_meta: Dict[int, ProjectMeta] = {}
        self.dataset_info: Dict[int, DatasetInfo] = {}
        self.video_info: Dict[int, VideoInfo] = {}
        self.children_datasets: Dict[int, List[DatasetInfo]] = {}
        self.children_items: Dict[int, List[Union[ImageInfo, VideoInfo]]] = {}

    def __getattr__(self, item: str) -> Dict:
        if not hasattr(self, item):
            new_dict = {}
            setattr(self, item, new_dict)
        return getattr(self, item)


class SamplingSettings:
    """Keys used in the sampling settings dict for `sample_video`."""

    ONLY_ANNOTATED = "annotated"
    START = "start"
    END = "end"
    STEP = "step"
    RESIZE = "resize"
    COPY_ANNOTATIONS = "copy_annotations"


def _get_frame_indices(frames_count, start, end, step, only_annotated, video_annotation):
    frame_indices = list(range(start, end if end is not None else frames_count))
    if only_annotated:
        annotated_frames = set()
        for frame in video_annotation.frames:
            frame: Frame
            if frame.figures:
                annotated_frames.add(frame.index)
        frame_indices = [idx for idx in frame_indices if idx in annotated_frames]
    frame_indices = [frame_indices[i] for i in range(0, len(frame_indices), step)]
    return frame_indices


def _frame_to_annotation(frame: Frame, video_annotation: VideoAnnotation) -> Annotation:

    labels = []
    for figure in frame.figures:
        tags = []
        video_object = figure.parent_object
        obj_class = video_object.obj_class
        geometry = figure.geometry
        for tag in video_object.tags:
            if tag.frame_range is None or tag.frame_range[0] <= frame.index <= tag.frame_range[1]:
                tags.append(Tag(tag.meta, tag.value, labeler_login=tag.labeler_login))
        tags.append(Tag(VIDEO_OBJECT_TAG_META, video_object.class_id))
        if figure.track_id is not None:
            tags.append(Tag(AUTO_TRACKED_TAG_META, None, labeler_login=video_object.labeler_login))
        label = Label(geometry, obj_class, TagCollection(tags))
        labels.append(label)
    img_tags = []
    for tag in video_annotation.tags:
        if tag.frame_range is None or tag.frame_range[0] <= frame.index <= tag.frame_range[1]:
            img_tags.append(Tag(tag.meta, tag.value, labeler_login=tag.labeler_login))
    return Annotation(video_annotation.img_size, labels=labels, img_tags=TagCollection(img_tags))


def _upload_annotations(api: Api, image_ids, frame_indices, video_annotation: VideoAnnotation):
    anns = []
    for frame_index in frame_indices:
        frame = video_annotation.frames.get(frame_index, None)
        if frame is not None:
            anns.append(_frame_to_annotation(frame, video_annotation))
        else:
            anns.append(Annotation(video_annotation.img_size))
    api.annotation.upload_anns(image_ids, anns=anns)


def _upload_frames(
    api: Api,
    frames: List[np.ndarray],
    video_name: str,
    video_frames_count: int,
    indices: List[int],
    dataset_id: int,
    sample_info: Dict = None,
    context: ApiContext = None,
    copy_annotations: bool = False,
    video_annotation: VideoAnnotation = None,
) -> List[int]:
    if sample_info is None:
        sample_info = {}
    if context is not None:
        if dataset_id not in context.children_items:
            context.children_items[dataset_id] = api.image.get_list(dataset_id)
        existing_images = context.children_items[dataset_id]
    else:
        existing_images = api.image.get_list(dataset_id)

    name_to_info = {image.name: image for image in existing_images}

    image_ids = [None] * len(frames)
    to_upload = []
    for i, index in enumerate(indices):
        digits = len(str(video_frames_count))
        image_name = f"{video_name}_frame_{str(index).zfill(digits)}.png"
        if image_name in name_to_info:
            image_ids[i] = name_to_info[image_name].id
        else:
            to_upload.append((image_name, i))

    if to_upload:
        frames = [frames[i] for _, i in to_upload]
        names = [name for name, _ in to_upload]
        metas = [{**sample_info, "frame_index": i} for _, i in to_upload]
        uploaded = api.image.upload_nps(
            dataset_id=dataset_id,
            names=names,
            imgs=frames,
            metas=metas,
        )

        for image_info, (_, i) in zip(uploaded, to_upload):
            image_ids[i] = image_info.id

    if copy_annotations:
        _upload_annotations(api, image_ids, indices, video_annotation)

    return image_ids


def sample_video(
    api: Api,
    video_id: int,
    dst_dataset_info: DatasetInfo,
    settings: Dict,
    sample_info: Dict = None,
    context: ApiContext = None,
    progress: tqdm_sly = None,
):
    dst_parent_info = dst_dataset_info
    only_annotated = settings.get(SamplingSettings.ONLY_ANNOTATED, False)
    start_frame = settings.get(SamplingSettings.START, 0)
    end_frame = settings.get(SamplingSettings.END, None)
    step = settings.get(SamplingSettings.STEP, 1)
    resize = settings.get(SamplingSettings.RESIZE, None)
    copy_annotations = settings.get(SamplingSettings.COPY_ANNOTATIONS, False)

    if context is None:
        context = ApiContext()
    if video_id not in context.video_info:
        context.video_info[video_id] = api.video.get_info_by_id(video_id)
    video_info = context.video_info[video_id]

    project_id = video_info.project_id
    if project_id not in context.project_meta:
        context.project_meta[project_id] = ProjectMeta.from_json(api.project.get_meta(project_id))
    project_meta = context.project_meta[project_id]

    video_annotation = VideoAnnotation.from_json(
        api.video.annotation.download(video_info.id), project_meta, key_id_map=KeyIdMap()
    )

    progress_cb = None
    if progress is not None:
        size = int(video_info.file_meta["size"])
        progress.reset(size)
        progress.unit = "B"
        progress.unit_scale = True
        progress.unit_divisor = 1024
        progress.message = f"Downloading {video_info.name} [{video_info.id}]"
        progress.desc = progress.message
        progress.refresh()
        progress_cb = progress.update

    video_path = f"/tmp/{video_info.name}"
    api.video.download_path(video_info.id, video_path, progress_cb=progress_cb)

    frame_indices = _get_frame_indices(
        video_info.frames_count, start_frame, end_frame, step, only_annotated, video_annotation
    )

    dst_dataset_info = _get_or_create_dst_dataset(
        api=api,
        src_info=video_info,
        dst_parent_info=dst_parent_info,
        sample_info=sample_info,
        context=context,
    )

    if progress is not None:
        progress.reset(len(frame_indices))
        progress.unit = "it"
        progress.unit_scale = False
        progress.unit_divisor = 1000
        progress.message = f"Processing {video_info.name} [{video_info.id}]"
        progress.desc = progress.message
        progress.miniters = 1
        progress.refresh()

    batch_size = 50
    try:
        with VideoFrameReader(video_path, frame_indices) as reader:
            for batch_indices in batched_iter(frame_indices, batch_size):
                batch_indices_list = list(batch_indices)
                frames = reader.read_batch(batch_indices_list)

                if resize:
                    resized_frames = []
                    for frame in frames:
                        resized_frame = cv2.resize(
                            frame,
                            (resize[1], resize[0]),  # (width, height)
                            interpolation=cv2.INTER_LINEAR,
                        )
                        resized_frames.append(resized_frame)
                    frames = resized_frames

                image_ids = _upload_frames(
                    api=api,
                    frames=frames,
                    video_name=video_info.name,
                    video_frames_count=video_info.frames_count,
                    indices=batch_indices_list,
                    dataset_id=dst_dataset_info.id,
                    sample_info=sample_info,
                    context=context,
                    copy_annotations=copy_annotations,
                    video_annotation=video_annotation,
                )

                if progress is not None:
                    progress.update(len(image_ids))

                # Free memory after each batch
                del frames
                if resize:
                    del resized_frames
    finally:
        import os
        if os.path.exists(video_path):
            os.remove(video_path)


def _get_or_create_dst_dataset(
    api: Api,
    src_info: Union[DatasetInfo, VideoInfo],
    dst_parent_info: Union[ProjectInfo, DatasetInfo],
    sample_info: Dict = None,
    context: ApiContext = None,
) -> DatasetInfo:
    if isinstance(dst_parent_info, ProjectInfo):
        parent = None
        project_id = dst_parent_info.id
    else:
        parent = dst_parent_info.id
        project_id = dst_parent_info.project_id

    if context is not None:
        if dst_parent_info.id not in context.children_datasets:
            context.children_datasets[dst_parent_info.id] = api.dataset.get_list(
                project_id, parent_id=parent
            )
        dst_datasets = context.children_datasets[dst_parent_info.id]
    else:
        dst_datasets = api.dataset.get_list(project_id, parent_id=parent)

    for dst_dataset in dst_datasets:
        if dst_dataset.name == src_info.name:
            return dst_dataset

    if isinstance(src_info, DatasetInfo):
        src_dataset_info = src_info
        description = (
            f"Sample dataset made from project #{src_info.parent_id}, dataset #{src_info.id}"
        )
    else:
        if context is not None:
            if src_info.dataset_id not in context.dataset_info:
                context.dataset_info[src_info.dataset_id] = api.dataset.get_info_by_id(
                    src_info.dataset_id
                )
            src_dataset_info = context.dataset_info[src_info.dataset_id]
        else:
            src_dataset_info = api.dataset.get_info_by_id(src_info.dataset_id)
        description = f"Sample dataset made from project #{src_info.project_id}, dataset #{src_info.dataset_id}, video #{src_info.id}"

    dst_dataset = api.dataset.create(
        project_id,
        name=src_info.name,
        description=description,
        parent_id=parent,
    )
    if sample_info is None:
        if context is not None:
            if src_info.project_id not in context.project_info:
                context.project_info[src_info.project_id] = api.project.get_info_by_id(
                    src_info.project_id
                )
            src_project_info = context.project_info[src_info.project_id]
        else:
            src_project_info = api.project.get_info_by_id(src_info.project_id)
        sample_info = {
            "is_sample": True,
            "video_project_id": src_info.project_id,
            "video_project_name": src_project_info.name,
        }
    sample_info.update(
        {
            "video_dataset_id": src_dataset_info.id,
            "video_dataset_name": src_dataset_info.name,
        }
    )
    if isinstance(src_info, VideoInfo):
        sample_info.update(
            {
                "video_id": src_info.id,
                "video_name": src_info.name,
            }
        )
    api.dataset.update_custom_data(
        dst_dataset.id,
        custom_data=sample_info,
    )
    if context is not None:
        context.dataset_info[dst_dataset.id] = dst_dataset
    return dst_dataset


def sample_video_dataset(
    api: Api,
    src_dataset_id: int,
    dst_parent_info: Union[ProjectInfo, DatasetInfo],
    settings: Dict,
    sample_info: Dict = None,
    context: ApiContext = None,
    datasets_ids_whitelist: List[int] = None,
    items_progress_cb: tqdm_sly = None,
    video_progress: tqdm_sly = None,
):
    if context is None:
        context = ApiContext()

    if not (
        datasets_ids_whitelist is None
        or src_dataset_id in datasets_ids_whitelist  # this dataset should be sampled
        or _has_children_datasets(
            api, src_dataset_id, context=context, children_ids=datasets_ids_whitelist
        )  # has children datasets that should be sampled
    ):
        return None

    src_dataset_info = context.dataset_info.get(src_dataset_id, None)
    if src_dataset_info is None:
        src_dataset_info = api.dataset.get_info_by_id(src_dataset_id)
        context.dataset_info[src_dataset_id] = src_dataset_info

    dst_dataset = _get_or_create_dst_dataset(
        api=api,
        src_info=src_dataset_info,
        dst_parent_info=dst_parent_info,
        sample_info=sample_info,
        context=context,
    )

    if datasets_ids_whitelist is None or src_dataset_id in datasets_ids_whitelist:
        video_infos = api.video.get_list(src_dataset_id)
        for video_info in video_infos:
            sample_video(
                api=api,
                video_id=video_info.id,
                dst_dataset_info=dst_dataset,
                settings=settings,
                sample_info=sample_info.copy(),
                context=context,
                progress=video_progress,
            )
            if items_progress_cb is not None:
                items_progress_cb()

    if src_dataset_id not in context.children_datasets:
        if src_dataset_id not in context.dataset_info:
            context.dataset_info[src_dataset_id] = api.dataset.get_info_by_id(src_dataset_id)
        src_dataset_info = context.dataset_info[src_dataset_id]
        context.children_datasets[src_dataset_id] = api.dataset.get_list(
            src_dataset_info.project_id, parent_id=src_dataset_info.id
        )
    datasets = context.children_datasets[src_dataset_id]
    for dataset in datasets:
        sample_video_dataset(
            api=api,
            src_dataset_id=dataset.id,
            dst_parent_info=dst_dataset,
            settings=settings,
            datasets_ids_whitelist=datasets_ids_whitelist,
            sample_info=sample_info,
            context=context,
            items_progress_cb=items_progress_cb,
            video_progress=video_progress,
        )
    return dst_dataset


def _update_meta(
    api: Api, src_project_meta: ProjectMeta, dst_project_id: int, context: ApiContext = None
):
    if context is None:
        context = ApiContext()
    if dst_project_id not in context.project_meta:
        context.project_meta[dst_project_id] = ProjectMeta.from_json(
            api.project.get_meta(dst_project_id)
        )
    dst_project_meta = context.project_meta[dst_project_id]
    dst_project_meta = dst_project_meta.merge(src_project_meta)
    if dst_project_meta.get_tag_meta(VIDEO_OBJECT_TAG_META.name) is None:
        dst_project_meta = dst_project_meta.add_tag_meta(VIDEO_OBJECT_TAG_META)
    if dst_project_meta.get_tag_meta(AUTO_TRACKED_TAG_META.name) is None:
        dst_project_meta = dst_project_meta.add_tag_meta(AUTO_TRACKED_TAG_META)

    if dst_project_meta != src_project_meta:
        dst_project_meta = api.project.update_meta(dst_project_id, dst_project_meta.to_json())
        context.project_meta[dst_project_id] = dst_project_meta


def _get_or_create_dst_project(
    api: Api,
    src_project_id: int,
    dst_project_id: Union[int, None] = None,
    sample_info: Dict = None,
    context: ApiContext = None,
) -> ProjectInfo:
    if dst_project_id is None:
        # get source project info
        if context is None:
            src_project_info = api.project.get_info_by_id(src_project_id)
        else:
            if src_project_id not in context.project_info:
                context.project_info[src_project_id] = api.project.get_info_by_id(src_project_id)
            src_project_info = context.project_info[src_project_id]
        # create new project
        if sample_info is None:
            sample_info = {}
        sample_info.update(
            {
                "is_sample": True,
                "video_project_id": src_project_id,
                "video_project_name": src_project_info.name,
            }
        )
        dst_project = api.project.create(
            src_project_info.workspace_id,
            f"{src_project_info.name}(images)",
            description=f"Sample project made from project #{src_project_info.id}",
            change_name_if_conflict=True,
        )
        api.project.update_custom_data(dst_project.id, sample_info)
    else:
        # use existing project
        dst_project = api.project.get_info_by_id(dst_project_id)
    if context is not None:
        context.project_info[dst_project.id] = dst_project
    if src_project_id not in context.project_meta:
        context.project_meta[src_project_id] = ProjectMeta.from_json(
            api.project.get_meta(src_project_id)
        )
    src_project_meta = context.project_meta[src_project_id]
    _update_meta(api, src_project_meta, dst_project.id, context=context)
    return dst_project


def _has_children_datasets(
    api: Api, dataset_id: int, context: ApiContext = None, children_ids: List[int] = None
) -> bool:
    if context is None:
        context = ApiContext()

    if dataset_id not in context.dataset_info:
        context.dataset_info[dataset_id] = api.dataset.get_info_by_id(dataset_id)
    dataset_info = context.dataset_info[dataset_id]
    if dataset_id not in context.children_datasets:
        context.children_datasets[dataset_id] = api.dataset.get_list(
            project_id=dataset_info.project_id, parent_id=dataset_id
        )

    if children_ids is None:
        return len(context.children_datasets[dataset_id]) > 0
    for child_dataset in context.children_datasets[dataset_id]:
        if child_dataset.id in children_ids:
            return True
        if _has_children_datasets(
            api, child_dataset.id, context=context, children_ids=children_ids
        ):
            return True
    return False


def sample_video_project(
    api: Api,
    project_id: int,
    settings: Dict,
    dst_project_id: Union[int, None] = None,
    datasets_ids: List[int] = None,
    context: ApiContext = None,
    items_progress_cb: tqdm_sly = None,
    video_progress: tqdm_sly = None,
):
    if context is None:
        context = ApiContext()

    if project_id not in context.project_info:
        context.project_info[project_id] = api.project.get_info_by_id(project_id)

    src_project_info = context.project_info[project_id]

    sample_info = {
        "is_sample": True,
        "video_project_id": src_project_info.id,
        "video_project_name": src_project_info.name,
    }
    dst_project_info = _get_or_create_dst_project(
        api, project_id, dst_project_id, sample_info, context
    )

    # non recursive. Nested datasets are handled by sample_video_dataset
    if project_id not in context.children_datasets:
        datasets = api.dataset.get_list(project_id)
        context.children_datasets[project_id] = datasets

    dataset_infos = context.children_datasets[project_id]
    for dataset_info in dataset_infos:
        context.dataset_info[dataset_info.id] = dataset_info
    for dataset_info in dataset_infos:
        sample_video_dataset(
            api=api,
            src_dataset_id=dataset_info.id,
            dst_parent_info=dst_project_info,
            settings=settings,
            sample_info=sample_info,
            context=context,
            datasets_ids_whitelist=datasets_ids,
            items_progress_cb=items_progress_cb,
            video_progress=video_progress,
        )

    return dst_project_info


def _extract_video_url(
    json_info: dict,
    video_id: int,
    server_address: str,
) -> str:
    """Extract and rewrite the public video URL from already-fetched video JSON info."""

    video_url = json_info.get(ApiField.FULL_STORAGE_URL)
    if not video_url:
        raise RuntimeError(
            f"Cannot resolve {ApiField.FULL_STORAGE_URL} for video {video_id}. "
            "Make sure the video is fully processed on the server."
        )
    parsed = urllib.parse.urlparse(video_url)
    public = urllib.parse.urlparse(server_address)
    return urllib.parse.urlunparse(parsed._replace(scheme=public.scheme, netloc=public.netloc))


def _resolve_video_url(api: Any, video_id: int) -> str:
    """Resolve and rewrite the public video URL for the current server address."""
    json_info = api.video.get_json_info_by_id(video_id, force_metadata_for_links=False)
    return _extract_video_url(
        json_info=json_info,
        video_id=video_id,
        server_address=api.server_address,
    )


def _av_open(video_url: str, token: Optional[str] = None) -> Any:
    """Open a PyAV container synchronously, injecting the API token header when present."""
    try:
        import av
    except ImportError as e:
        raise ImportError(
            "PyAV is required for video frame streaming but is not installed. "
            "Install it with: pip install 'supervisely[video-av]'"
        ) from e

    av_options: Dict[str, str] = {
        # FFmpeg read/write timeout in microseconds (30 s).
        # Without this, av.open / container.decode can block the worker thread
        # indefinitely when the remote server stalls or the connection drops.
        "rw_timeout": "30000000",
    }
    if token:
        av_options["headers"] = f"x-api-key: {token}\r\n"
    return av.open(video_url, options=av_options)


def _sync_build_pts_map(
    video_url: str,
    token: Optional[str],
    video_id: int,
    end_frame: Optional[int] = None,
) -> Tuple[List[int], bool]:
    """Synchronously demux video and build a sorted PTS list. Safe to run in a thread pool.

    :param video_url: Public URL of the video to demux.
    :param token: API token for authenticated access to the video URL, if required.
    :param video_id: Video ID (for logging purposes only).
    :param end_frame: Stop demuxing after collecting ``end_frame + 64`` PTS values.
        Avoids scanning the entire video when only a subset of frames is needed.
        Early termination happens at ``end_frame + 64`` instead of ``end_frame`` to account for B-frame reordering (typically ≤16 frames).
        ``None`` reads the full video (required when ``end`` is unknown).
    """
    container = _av_open(video_url, token)
    try:
        video_streams = container.streams.video
        if not video_streams:
            raise RuntimeError(f"PyAV found no video streams in video {video_id}.")
        v_stream = video_streams[0]

        pts_list = []
        saw_pts_ne_dts = False
        stop_after = (end_frame + 64) if end_frame is not None else None
        for pkt in container.demux(v_stream):
            if pkt.pts is not None and pkt.dts is not None and pkt.pts != pkt.dts:
                saw_pts_ne_dts = True
            if pkt.pts is not None and pkt.pts >= 0:
                pts_list.append(pkt.pts)
            if stop_after is not None and len(pts_list) >= stop_after:
                break

        pts_list.sort()
        has_b_frames = getattr(v_stream.codec_context, "has_b_frames", 0) > 0
        decode_from_start = has_b_frames and not saw_pts_ne_dts
        return pts_list, decode_from_start
    finally:
        container.close()


async def async_stream_video_frames(
    api: Any,
    video_id: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    queue_maxsize: int = 32,
) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
    """Async generator that streams decoded video frames by frame index.

    Builds a PTS map by demuxing the video, then decodes and yields frames
    in the requested range. Automatically selects the optimal decoding
    strategy: seek-based for most videos, or full decode from start for
    B-frame streams without a CTTS box.

    :param api: Supervisely API object.
    :type api: Api
    :param video_id: Video ID in Supervisely.
    :type video_id: int
    :param start: First frame index to yield (inclusive, 0-based). Defaults to 0.
    :type start: int, optional
    :param end: Last frame index to yield (inclusive, 0-based). Defaults to the last frame.
    :type end: int, optional
    :param queue_maxsize: Decode prefetch buffer size in frames (default 32).
        Increase on memory-rich systems for higher throughput.
        Peak RAM ≈ ``(queue_maxsize + 1) x H x W x 3`` bytes.
    :type queue_maxsize: int, optional
    :yields: Tuple of ``(frame_index, rgb_image)`` where ``rgb_image`` is a
        ``numpy.ndarray`` of shape ``(H, W, 3)`` in RGB uint8 format.
    :rtype: AsyncGenerator[Tuple[int, numpy.ndarray], None]

    :Usage Example:

        .. code-block:: python

            import supervisely as sly
            from supervisely.video.sampling import async_stream_video_frames

            api = sly.Api.from_env()

            async for frame_idx, img in async_stream_video_frames(api, video_id=123, start=0, end=9):
                print(frame_idx, img.shape)
    """

    video_url = _resolve_video_url(api, video_id)
    loop = asyncio.get_running_loop()

    decode_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="av_decode"
    )

    try:
        pts_list, decode_from_start = await loop.run_in_executor(
            decode_executor, _sync_build_pts_map, video_url, api.token, video_id, end
        )
    except Exception:
        decode_executor.shutdown(wait=False)
        raise

    if decode_from_start:
        logger.warning(
            "B-frame stream without CTTS detected; "
            "using full decode from start for stable frame extraction.",
            extra={"video_id": video_id},
        )
    logger.debug(
        "build_pts_map: video_id=%d - %d pts values (demux, non-negative, sorted), decode_from_start=%s.",
        video_id,
        len(pts_list),
        decode_from_start,
    )

    if not pts_list:
        decode_executor.shutdown(wait=False)
        return

    _start = max(0, start if start is not None else 0)
    _end = min(max(_start, end), len(pts_list) - 1) if end is not None else len(pts_list) - 1

    pts_to_index = {pts: idx for idx, pts in enumerate(pts_list)}

    queue = asyncio.Queue(maxsize=queue_maxsize)
    cancel_event = threading.Event()

    def _put(item: Any) -> bool:
        while not cancel_event.is_set():
            fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            try:
                fut.result(timeout=0.2)
                return True
            except concurrent.futures.TimeoutError:
                continue
        return False

    def _worker() -> None:
        try:
            container: Any = _av_open(video_url, api.token)
            try:
                video_streams = container.streams.video
                if not video_streams:
                    raise RuntimeError(f"PyAV found no video streams in video {video_id}.")
                v_stream = video_streams[0]

                if decode_from_start:
                    remaining: set = set(range(_start, _end + 1))
                    for pkt in container.demux(v_stream):
                        if cancel_event.is_set():
                            break
                        for frame in pkt.decode():
                            if frame.pts is None or frame.is_corrupt:
                                continue
                            frame_idx = pts_to_index.get(frame.pts)
                            if frame_idx is None or frame_idx not in remaining:
                                continue
                            img = frame.to_ndarray(format="rgb24")
                            if not _put((frame_idx, img)):
                                return
                            remaining.discard(frame_idx)
                        if not remaining:
                            break
                else:
                    # Single seek to the first requested frame, then decode sequentially.
                    # This avoids O(N) seeks for contiguous ranges.
                    wanted: Dict[int, int] = {pts_list[i]: i for i in range(_start, _end + 1)}
                    end_pts: int = pts_list[_end]
                    container.seek(
                        pts_list[_start], stream=v_stream, backward=True, any_frame=False
                    )
                    for frame in container.decode(v_stream):
                        if cancel_event.is_set():
                            break
                        if frame.pts is None or frame.is_corrupt:
                            continue
                        frame_idx = wanted.pop(frame.pts, None)
                        if frame_idx is not None:
                            img = frame.to_ndarray(format="rgb24")
                            if not _put((frame_idx, img)):
                                return
                        if not wanted:
                            break
                        if frame.pts > end_pts:
                            break
            finally:
                container.close()
        except Exception as exc:
            _put(exc)
        finally:
            _put(None)  # sentinel — always sent last

    future = loop.run_in_executor(decode_executor, _worker)
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        cancel_event.set()
        await future
        decode_executor.shutdown(wait=True)


async def async_stream_video_frames_to_dir(
    api: Any,
    video_id: int,
    output_dir: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    ext: str = "png",
    progress_cb: Optional[Callable] = None,
    image_writer: Optional[Callable[[str, np.ndarray], None]] = None,
    max_write_workers: int = 4,
    queue_maxsize: int = 32,
) -> List[str]:
    """Async version of :func:`stream_video_frames_to_dir`.

    Streams decoded video frames and saves them to ``output_dir`` as image
    files named ``frame_<index:06d>.<ext>``.

    :param api: Supervisely API object.
    :type api: Api
    :param video_id: Video ID in Supervisely.
    :type video_id: int
    :param output_dir: Directory where frame images will be saved.
    :type output_dir: str
    :param start: First frame index to save (inclusive, 0-based). Defaults to 0.
    :type start: int, optional
    :param end: Last frame index to save (inclusive, 0-based). Defaults to the last frame.
    :type end: int, optional
    :param ext: Image file extension (e.g. ``"png"``, ``"jpg"``). Defaults to ``"png"``.
    :type ext: str, optional
    :param progress_cb: Callable invoked with ``1`` after each frame is saved.
    :type progress_cb: callable, optional
    :param image_writer: Custom function ``(path, image) -> None`` for writing frames.
        Defaults to :func:`supervisely.imaging.image.write`.
    :type image_writer: callable, optional
    :param max_write_workers: Number of parallel write threads (default 4).
    :type max_write_workers: int, optional
    :param queue_maxsize: Decode prefetch buffer size in frames (default 32).
    :type queue_maxsize: int, optional
    :returns: List of absolute paths to saved frame files.
    :rtype: List[str]
    """
    os.makedirs(output_dir, exist_ok=True)
    if image_writer is None:
        from supervisely.imaging import image as sly_image

        image_writer = sly_image.write

    loop = asyncio.get_running_loop()
    # Semaphore bounds concurrent writes in flight — primary OOM guard.
    # When all slots are occupied, acquire() blocks the consumer loop,
    # which fills the decode queue, which stalls the FFmpeg worker.
    write_sem = asyncio.Semaphore(max_write_workers)
    write_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_write_workers, thread_name_prefix="frame_write"
    )
    saved_paths = []
    pending = []

    async def _write_frame(path: str, img: np.ndarray) -> None:
        try:
            await loop.run_in_executor(write_executor, image_writer, path, img)
        finally:
            write_sem.release()

    try:
        async for frame_idx, img in async_stream_video_frames(
            api=api,
            video_id=video_id,
            start=start,
            end=end,
            queue_maxsize=queue_maxsize,
        ):
            path = os.path.join(output_dir, f"frame_{frame_idx:06d}.{ext}")
            await write_sem.acquire()
            saved_paths.append(path)
            pending.append(asyncio.create_task(_write_frame(path, img)))
            if progress_cb is not None:
                progress_cb(1)

        if pending:
            await asyncio.gather(*pending)
    except Exception:
        for task in pending:
            task.cancel()
        raise
    finally:
        write_executor.shutdown(wait=False)

    return saved_paths


def stream_video_frames_to_dir(
    api: Any,
    video_id: int,
    output_dir: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    ext: str = "png",
    progress_cb: Optional[Callable] = None,
    image_writer: Optional[Callable[[str, np.ndarray], None]] = None,
    max_write_workers: int = 4,
    queue_maxsize: int = 32,
) -> List[str]:
    """Stream decoded video frames and save them to a directory.

    Decodes the requested frame range from the video (identified by
    ``video_id``) and writes each frame as an image file to ``output_dir``.
    Files are named ``frame_<index:06d>.<ext>`` (e.g. ``frame_000162.png``).

    :param api: Supervisely API object.
    :type api: Api
    :param video_id: Video ID in Supervisely.
    :type video_id: int
    :param output_dir: Directory where frame images will be saved.
        Created automatically if it does not exist.
    :type output_dir: str
    :param start: First frame index to save (inclusive, 0-based). Defaults to 0.
    :type start: int, optional
    :param end: Last frame index to save (inclusive, 0-based). Defaults to the last frame.
    :type end: int, optional
    :param ext: Image file extension (e.g. ``"png"``, ``"jpg"``). Defaults to ``"png"``.
    :type ext: str, optional
    :param progress_cb: Callable invoked with ``1`` after each frame is saved.
        Useful for progress bars.
    :type progress_cb: callable, optional
    :param image_writer: Custom function ``(path, image) -> None`` for writing frames.
        Defaults to :func:`supervisely.imaging.image.write`.
    :type image_writer: callable, optional
    :param max_write_workers: Number of parallel write threads (default 4).
        Increase on memory-rich systems for higher write throughput.
    :type max_write_workers: int, optional
    :param queue_maxsize: Decode prefetch buffer size in frames (default 32).
        Peak RAM ≈ ``(queue_maxsize + max_write_workers + 1) x frame_bytes``.
    :type queue_maxsize: int, optional
    :returns: List of absolute paths to the saved frame files.
    :rtype: List[str]

    :Usage Example:

        .. code-block:: python

            import supervisely as sly
            from supervisely.video.sampling import stream_video_frames_to_dir

            api = sly.Api.from_env()

            paths = stream_video_frames_to_dir(
                api, video_id=123, output_dir="/tmp/frames", start=0, end=9
            )
            print(paths)  # ['/tmp/frames/frame_000000.png', ...]
    """
    coro = async_stream_video_frames_to_dir(
        api=api,
        video_id=video_id,
        output_dir=output_dir,
        start=start,
        end=end,
        ext=ext,
        progress_cb=progress_cb,
        image_writer=image_writer,
        max_write_workers=max_write_workers,
        queue_maxsize=queue_maxsize,
    )
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None and running_loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return run_coroutine(coro)
