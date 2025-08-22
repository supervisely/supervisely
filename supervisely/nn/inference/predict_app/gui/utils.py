from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from supervisely import logger
from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.api.dataset_api import DatasetInfo
from supervisely.project.project import ProjectType
from supervisely.app.widgets import Progress
from supervisely.app import DataJson
from supervisely.app.widgets import Button, Card, Stepper, Text, Widget

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

    def button_click(button_clicked_value: Optional[bool] = None):
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
            if on_select_click is not None:
                for func in on_select_click:
                    func()
        else:
            update_custom_button_params(button, select_params)
            if on_reselect_click is not None:
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
            callback(False)

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


def copy_project(
    api: Api,
    project_name: str,
    workspace_id: int,
    project_id: int,
    dataset_ids: List[int] = [],
    with_annotations: bool = True,
    progress: Progress = None,
):
    """
    Copy a project

    :param api: Supervisely API
    :type api: Api
    :param project_name: Name of the new project
    :type project_name: str
    :param workspace_id: ID of the workspace
    :type workspace_id: int
    :param project_id: ID of the project to copy
    :type project_id: int
    :param dataset_ids: List of dataset IDs to copy. If empty, all datasets from the project will be copied.
    :type dataset_ids: List[int]
    :param with_annotations: Whether to copy annotations
    :type with_annotations: bool
    :param progress: Progress callback
    :type progress: Progress
    :return: Created project
    :rtype: ProjectInfo
    """

    def _create_project() -> ProjectInfo:
        created_project = api.project.create(
            workspace_id,
            project_name,
            type=ProjectType.IMAGES,
            change_name_if_conflict=True,
        )
        if with_annotations:
            api.project.merge_metas(src_project_id=project_id, dst_project_id=created_project.id)
        return created_project

    def _copy_full_project(
        created_project: ProjectInfo, src_datasets_tree: Dict[DatasetInfo, Dict]
    ):
        src_dst_ds_id_map: Dict[int, int] = {}

        def _create_full_tree(ds_tree: Dict[DatasetInfo, Dict], parent_id: int = None):
            for src_ds, nested_src_ds_tree in ds_tree.items():
                dst_ds = api.dataset.create(
                    project_id=created_project.id,
                    name=src_ds.name,
                    description=src_ds.description,
                    change_name_if_conflict=True,
                    parent_id=parent_id,
                )
                src_dst_ds_id_map[src_ds.id] = dst_ds

                # Preserve dataset custom data
                info_ds = api.dataset.get_info_by_id(src_ds.id)
                if info_ds.custom_data:
                    api.dataset.update_custom_data(dst_ds.id, info_ds.custom_data)
                _create_full_tree(nested_src_ds_tree, parent_id=dst_ds.id)

        _create_full_tree(src_datasets_tree)

        for src_ds_id, dst_ds in src_dst_ds_id_map.items():
            _copy_items(src_ds_id, dst_ds)

    def _copy_datasets(created_project: ProjectInfo, src_datasets_tree: Dict[DatasetInfo, Dict]):
        created_datasets: Dict[int, DatasetInfo] = {}
        processed_copy: Set[int] = set()

        for dataset_id in dataset_ids:
            chain = find_parents_in_tree(src_datasets_tree, dataset_id, with_self=True)
            if not chain:
                logger.warning(
                    f"Dataset id {dataset_id} not found in project {project_id}. Skipping."
                )
                continue

            parent_created_id = None
            for ds_info in chain:
                if ds_info.id in created_datasets:
                    parent_created_id = created_datasets[ds_info.id].id
                    continue

                created_ds = api.dataset.create(
                    created_project.id,
                    ds_info.name,
                    description=ds_info.description,
                    change_name_if_conflict=False,
                    parent_id=parent_created_id,
                )
                created_datasets[ds_info.id] = created_ds
                src_info = api.dataset.get_info_by_id(ds_info.id)
                if src_info.custom_data:
                    api.dataset.update_custom_data(created_ds.id, src_info.custom_data)
                parent_created_id = created_ds.id

            if dataset_id not in processed_copy:
                _copy_items(dataset_id, created_datasets[dataset_id])
                processed_copy.add(dataset_id)

    def _copy_items(src_ds_id: int, dst_ds: DatasetInfo):
        input_img_infos = api.image.get_list(src_ds_id)
        with progress(
            message=f"Copying items from dataset: {dst_ds.name}", total=len(input_img_infos)
        ) as pbar:
            progress.show()
            api.image.copy_batch_optimized(
                src_dataset_id=src_ds_id,
                src_image_infos=input_img_infos,
                dst_dataset_id=dst_ds.id,
                with_annotations=with_annotations,
                progress_cb=pbar.update,
            )
            progress.hide()

    created_project = _create_project()
    src_datasets_tree = api.dataset.get_tree(project_id)

    if not dataset_ids:
        _copy_full_project(created_project, src_datasets_tree)
    else:
        _copy_datasets(created_project, src_datasets_tree)
    return created_project
