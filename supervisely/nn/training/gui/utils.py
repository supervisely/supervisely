from typing import List, Optional, Callable, Dict, Any
from supervisely.app.widgets import Button, Card, Stepper, Widget, Text
from supervisely.app import DataJson

from supervisely.nn.training.gui.input_selector import InputSelector
from supervisely.nn.training.gui.classes_selector import ClassesSelector

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


def wrap_button_click(
    button: Button,
    cards_to_unlock: List[Card],
    widgets_to_disable: List[Widget],
    callback: Optional[Callable] = None,
    lock_msg: str = None,
    upd_params: bool = True,
    validation_text: Text = None,
    validation_func: Optional[Callable] = None,
    on_button_click: Optional[Callable] = None,
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
            if on_button_click is not None:
                on_button_click()
        else:
            update_custom_button_params(button, select_params)
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

    return button_click


def set_stepper_step(stepper: Stepper, button: Button, next_pos: int):
    bid = button.widget_id
    if button_clicked[bid] is True:
        stepper.set_active_step(next_pos)
    else:
        stepper.set_active_step(next_pos - 1)

