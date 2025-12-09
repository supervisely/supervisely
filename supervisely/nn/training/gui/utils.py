from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from supervisely import Api
from supervisely.api.app_api import AppInfo
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


def get_module_info_by_name(api: Api, app_name: str) -> Union[Dict, None]:
    all_modules = api.app.get_list_ecosystem_modules()
    for module in all_modules:
        if module["name"] == app_name:
            app_info = api.app.get_info(module["id"])
            return app_info


def generate_task_check_function_js(folder: str) -> str:
    """
    Returns JavaScript function code for checking existing tasks.

    :param folder: Remote folder to check.
    :type folder: str
    :return: JavaScript function code for checking existing tasks.
    :rtype: str
    """
    escaped_folder = folder.replace("'", "\\'")
    js_code = f"""
        if (!task || !task.meta || !task.meta.params || !task.meta.params.state) {{
            return false;
        }}
        const taskFolder = task.meta.params.state.slyFolder;
        if (!taskFolder || typeof taskFolder !== 'string') {{
            return false;
        }}
        return taskFolder === '{escaped_folder}';
    """
    return js_code
