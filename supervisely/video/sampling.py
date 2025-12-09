from typing import Dict, List, Union

import cv2
import numpy as np

from supervisely._utils import batched_iter
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_collection import TagCollection
from supervisely.annotation.tag_meta import TagApplicableTo, TagMeta, TagValueType
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.image_api import ImageInfo
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
