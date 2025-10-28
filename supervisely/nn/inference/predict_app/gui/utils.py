from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from supervisely import logger
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.image_api import ImageInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.app import DataJson
from supervisely.app.widgets import Button, Card, Progress, Stepper, Text, Widget
from supervisely.nn.model.prediction import Prediction
from supervisely.project.project import ProjectType
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.video_project import VideoInfo
from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.video_annotation import VideoAnnotation
from supervisely.video_annotation.video_figure import VideoFigure
from supervisely.video_annotation.video_object import VideoObject
from supervisely.video_annotation.video_object_collection import VideoObjectCollection

button_clicked = {}


def update_custom_params(
    button: Button,
    params_dct: Dict[str, Any],
) -> None:
    button_state = button.get_json_data()
    for key in params_dct.keys():
        if key not in button_state:
            raise AttributeError(f"Parameter {key} doesn't exists.")
        else:
            DataJson()[button.widget_id][key] = params_dct[key]
    DataJson().send_changes()


def update_custom_button_params(
    button: Button,
    params_dct: Dict[str, Any],
) -> None:
    params = params_dct.copy()
    if "icon" in params and params["icon"] is not None:
        new_icon = f'<i class="{params["icon"]}" style="margin-right: {button._icon_gap}px"></i>'
        params["icon"] = new_icon
    update_custom_params(button, params)


def disable_enable(widgets: List[Widget], disable: bool = True):
    for w in widgets:
        if disable:
            w.disable()
        else:
            w.enable()


def unlock_lock(cards: List[Card], unlock: bool = True, message: str = None):
    for w in cards:
        if unlock:
            w.unlock()
            # w.uncollapse()
        else:
            w.lock(message)
            # w.collapse()


def collapse_uncollapse(cards: List[Card], collapse: bool = True):
    for w in cards:
        if collapse:
            w.collapse()
        else:
            w.uncollapse()


def wrap_button_click(
    button: Button,
    cards_to_unlock: List[Card],
    widgets_to_disable: List[Widget],
    callback: Optional[Callable] = None,
    lock_msg: str = None,
    upd_params: bool = True,
    validation_text: Text = None,
    validation_func: Optional[Callable] = None,
    on_select_click: Optional[Callable] = None,
    on_reselect_click: Optional[Callable] = None,
    collapse_card: Tuple[Card, bool] = None,
) -> Callable[[Optional[bool]], None]:
    global button_clicked

    select_params = {"icon": None, "plain": False, "text": "Select"}
    reselect_params = {"icon": "zmdi zmdi-refresh", "plain": True, "text": "Reselect"}
    bid = button.widget_id
    button_clicked[bid] = False

    def button_click(button_clicked_value: Optional[bool] = None, suppress_actions: bool = False):
        if button_clicked_value is None or button_clicked_value is False:
            if validation_func is not None:
                success = validation_func()
                if not success:
                    return

        if button_clicked_value is not None:
            button_clicked[bid] = button_clicked_value
        else:
            button_clicked[bid] = not button_clicked[bid]

        if button_clicked[bid] and upd_params:
            update_custom_button_params(button, reselect_params)
            if not suppress_actions and on_select_click is not None:
                for func in on_select_click:
                    func()
        else:
            update_custom_button_params(button, select_params)
            if not suppress_actions and on_reselect_click is not None:
                for func in on_reselect_click:
                    func()
            validation_text.hide()

        unlock_lock(
            cards_to_unlock,
            unlock=button_clicked[bid],
            message=lock_msg,
        )
        disable_enable(
            widgets_to_disable,
            disable=button_clicked[bid],
        )
        if callback is not None and not button_clicked[bid]:
            callback(False, True)

        if collapse_card is not None:
            card, collapse = collapse_card
            if collapse:
                collapse_uncollapse([card], collapse)

    return button_click


def set_stepper_step(stepper: Stepper, button: Button, next_pos: int):
    bid = button.widget_id
    if button_clicked[bid] is True:
        stepper.set_active_step(next_pos)
    else:
        stepper.set_active_step(next_pos - 1)


def find_parents_in_tree(
    tree: Dict[DatasetInfo, Dict], dataset_id: int, with_self: bool = False
) -> Optional[List[DatasetInfo]]:
    """
    Find all parent datasets in the tree for a given dataset ID.
    """

    def _dfs(subtree: Dict[DatasetInfo, Dict], parents: List[DatasetInfo]):
        for dataset_info, children in subtree.items():
            if dataset_info.id == dataset_id:
                if with_self:
                    return parents + [dataset_info]
                return parents
            res = _dfs(children, parents + [dataset_info])
            if res is not None:
                return res
        return None

    return _dfs(tree, [])


def _copy_items_to_dataset(
    api: Api,
    src_dataset_id: int,
    dst_dataset: DatasetInfo,
    project_type: str,
    with_annotations: bool = True,
    progress_cb: Callable = None,
    progress: Progress = None,
    items_infos: List[Union[ImageInfo, VideoInfo]] = None,
) -> Union[List[ImageInfo], List[VideoInfo]]:
    if progress is None:
        progress = Progress()

    def combined_progress(n):
        progress_cb(n)
        pbar.update(n)

    if project_type == ProjectType.IMAGES:
        if items_infos is None:
            items_infos = api.image.get_list(src_dataset_id)
        with progress(
            message=f"Copying items from dataset: {dst_dataset.name}", total=len(items_infos)
        ) as pbar:

            if progress_cb:
                _progress_cb = combined_progress
            else:
                _progress_cb = pbar.update

            progress.show()
            copied = api.image.copy_batch_optimized(
                src_dataset_id=src_dataset_id,
                src_image_infos=items_infos,
                dst_dataset_id=dst_dataset.id,
                with_annotations=with_annotations,
                progress_cb=_progress_cb,
            )
            progress.hide()
    elif project_type == ProjectType.VIDEOS:
        if items_infos is None:
            items_infos = api.video.get_list(src_dataset_id)

        with progress(
            message=f"Copying items from dataset: {dst_dataset.name}", total=len(items_infos)
        ) as pbar:
            if progress_cb:
                _progress_cb = combined_progress
            else:
                _progress_cb = pbar.update
            progress.show()
            copied = api.video.copy_batch(
                dst_dataset_id=dst_dataset.id,
                ids=[info.id for info in items_infos],
                with_annotations=with_annotations,
                progress_cb=_progress_cb,
            )
            progress.hide()
    else:
        raise NotImplementedError(f"Copy not implemented for project type {project_type}")
    return copied


def get_items_infos(
    api: Api, items_ids: List[int], project_type: str
) -> List[Union[ImageInfo, VideoInfo]]:
    if project_type == ProjectType.IMAGES:
        items_infos: List[ImageInfo] = api.image.get_info_by_id_batch(items_ids)
    elif project_type == ProjectType.VIDEOS:
        items_infos: List[VideoInfo] = api.video.get_info_by_id_batch(items_ids)
    else:
        raise NotImplementedError(f"Items of type {project_type} are not supported")
    return items_infos


def copy_items_to_project(
    api: Api,
    src_project_id: int,
    items: Union[List[ImageInfo], List[VideoInfo]],
    dst_project_id: int,
    with_annotations: bool = True,
    progress_cb: Progress = None,
    ds_progress: Progress = None,
    project_type: str = None,
    src_datasets_tree: Dict[DatasetInfo, Dict] = None,
) -> Union[List[ImageInfo], List[VideoInfo]]:
    if project_type is None:
        dst_project_info = api.project.get_info_by_id(src_project_id)
        project_type = dst_project_info.type
    if len(items) == 0:
        return []
    if len(set(info.project_id for info in items)) != 1:
        raise ValueError("Items must belong to the same project")

    items_by_dataset: Dict[int, List[Union[ImageInfo, VideoInfo]]] = {}
    for item_info in items:
        items_by_dataset.setdefault(item_info.dataset_id, []).append(item_info)

    if src_datasets_tree is None:
        src_datasets_tree = api.dataset.get_tree(src_project_id)

    created_datasets: Dict[int, DatasetInfo] = {}
    processed_copy: Set[int] = set()

    copied_items = {}
    for dataset_id, items_infos in items_by_dataset.items():
        chain = find_parents_in_tree(src_datasets_tree, dataset_id, with_self=True)
        if not chain:
            logger.warning(f"Dataset id {dataset_id} not found in project. Skipping")
            continue

        parent_created_id = None
        for ds_info in chain:
            if ds_info.id in created_datasets:
                parent_created_id = created_datasets[ds_info.id].id
                continue

            created_ds = api.dataset.create(
                dst_project_id,
                ds_info.name,
                description=ds_info.description,
                change_name_if_conflict=False,
                parent_id=parent_created_id,
            )
            if ds_info.custom_data:
                created_ds = api.dataset.update_custom_data(created_ds.id, ds_info.custom_data)
            created_datasets[ds_info.id] = created_ds
            parent_created_id = created_ds.id

        if dataset_id not in processed_copy:
            copied_ds_items = _copy_items_to_dataset(
                api=api,
                src_dataset_id=dataset_id,
                dst_dataset=created_datasets[dataset_id],
                project_type=project_type,
                with_annotations=with_annotations,
                progress_cb=progress_cb,
                progress=ds_progress,
                items_infos=items_infos,
            )
            for src_info, dst_info in zip(items_infos, copied_ds_items):
                copied_items[src_info.id] = dst_info
            processed_copy.add(dataset_id)
    return [copied_items[item.id] for item in items]


def create_project(
    api: Api,
    project_id: int,
    project_name: str,
    workspace_id: int,
    copy_meta: bool = False,
    project_type: str = None,
) -> ProjectInfo:
    if project_type is None:
        project_info = api.project.get_info_by_id(project_id)
        project_type = project_info.type
    created_project = api.project.create(
        workspace_id,
        project_name,
        type=project_type,
        change_name_if_conflict=True,
    )
    if copy_meta:
        api.project.merge_metas(src_project_id=project_id, dst_project_id=created_project.id)
    return created_project


def copy_project(
    api: Api,
    project_id: int,
    workspace_id: int,
    project_name: str,
    items_ids: List[int] = None,
    with_annotations: bool = True,
    progress: Progress = None,
) -> ProjectInfo:
    dst_project = create_project(
        api, project_id, project_name, workspace_id=workspace_id, copy_meta=True
    )
    items = []
    if items_ids is None:
        project_type = dst_project.type
        datasets = api.dataset.get_list(project_id, recursive=True)
        if project_type == ProjectType.IMAGES:
            get_items_f = api.image.get_list
        elif project_type == ProjectType.VIDEOS:
            get_items_f = api.video.get_list
        else:
            raise NotImplementedError(f"Project type {project_type} is not supported")
        for ds in datasets:
            ds_items = get_items_f(dataset_id=ds.id)
            if ds_items:
                items.extend(ds_items)
    else:
        items = get_items_infos(api, items_ids, dst_project.type)
    copy_items_to_project(
        api=api,
        src_project_id=project_id,
        items=items,
        dst_project_id=dst_project.id,
        with_annotations=with_annotations,
        ds_progress=progress,
        project_type=dst_project.type,
    )
    return dst_project


def video_annotation_from_predictions(
    predictions: List[Prediction], project_meta: ProjectMeta, frame_size: Tuple[int, int]
) -> VideoAnnotation:
    objects = {}
    frames = []
    for i, prediction in enumerate(predictions):
        figures = []
        for label in prediction.annotation.labels:
            obj_name = label.obj_class.name
            if not obj_name in objects:
                obj_class = project_meta.get_obj_class(obj_name)
                if obj_class is None:
                    continue
                objects[obj_name] = VideoObject(obj_class)

            vid_object = objects[obj_name]
            if vid_object:
                figures.append(VideoFigure(vid_object, label.geometry, frame_index=i))
        frame = Frame(i, figures=figures)
        frames.append(frame)
    return VideoAnnotation(
        img_size=frame_size,
        frames_count=len(frames),
        objects=VideoObjectCollection(list(objects.values())),
        frames=FrameCollection(frames),
    )
