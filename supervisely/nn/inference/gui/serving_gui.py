from functools import wraps
from typing import Callable, Dict, List, Optional, Union

import yaml

import supervisely.app.widgets as Widgets
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress


class ServingGUI:
    def __init__(self) -> None:
        self._device_select = Widgets.SelectCudaDevice(include_cpu_option=True)
        self._device_field = Widgets.Field(self._device_select, title="Device")
        self._serve_button = Widgets.Button("SERVE")
        self._success_label = Widgets.DoneLabel()
        self._success_label.hide()
        self._download_progress = Widgets.Progress("Downloading model...", hide_on_finish=True)
        self._download_progress.hide()
        self._change_model_button = Widgets.Button(
            "STOP AND CHOOSE ANOTHER MODEL", button_type="danger"
        )
        self._change_model_button.hide()

        self.serve_container = Widgets.Container(
            [
                self._device_field,
                self._download_progress,
                self._success_label,
                self._serve_button,
                self._change_model_button,
            ],
        )
        self.serve_model_card = Widgets.Card(
            title="Serve Model",
            description="Download and deploy the model on the selected device.",
            content=self.serve_container,
        )

        self._model_inference_settings_widget = Widgets.Editor(
            readonly=True, restore_default_button=False
        )
        self._model_inference_settings_container = Widgets.Field(
            self._model_inference_settings_widget,
            title="Inference settings",
            description="Model allows user to configure the following parameters on prediction phase",
        )

        self._model_info_widget = Widgets.ModelInfo()
        self._model_info_widget_container = Widgets.Field(
            self._model_info_widget,
            title="Session Info",
            description="Basic information about the deployed model",
        )

        self._model_classes_widget = Widgets.ClassesTable(selectable=False)
        self._model_classes_plug = Widgets.Text("No classes provided")
        self._model_classes_widget_container = Widgets.Field(
            content=Widgets.Container([self._model_classes_widget, self._model_classes_plug]),
            title="Model classes",
            description="List of classes model predicts",
        )

        self._model_full_info = Widgets.Container(
            [
                self._model_info_widget_container,
                self._model_inference_settings_container,
                self._model_classes_widget_container,
            ]
        )
        self._model_full_info.hide()
        self._before_deploy_msg = Widgets.Text("Deploy model to see the information.")

        self._model_full_info_card = Widgets.Card(
            title="Full model info",
            description="Inference settings, session parameters and model classes",
            collapsable=True,
            content=Widgets.Container(
                [
                    self._model_full_info,
                    self._before_deploy_msg,
                ]
            ),
        )

        self._model_full_info_card.collapse()
        self._additional_ui_content = []
        self.get_ui = self.__add_content_and_model_info_to_default_ui(
            self._model_full_info_card
        )  # pylint: disable=method-hidden

        self.on_change_model_callbacks: List[Callable] = [ServingGUI._hide_info_after_change]
        self.on_serve_callbacks: List[Callable] = []

        @self.serve_button.click
        def serve_model():
            self.deploy_with_current_params()

        @self._change_model_button.click
        def change_model():
            for cb in self.on_change_model_callbacks:
                cb(self)
            self.change_model()

    @property
    def serve_button(self) -> Widgets.Button:
        return self._serve_button

    @property
    def download_progress(self) -> Widgets.Progress:
        return self._download_progress

    @property
    def device(self) -> str:
        return self._device_select.get_device()

    def get_device(self) -> str:
        return self._device_select.get_device()

    def deploy_with_current_params(self):
        for cb in self.on_serve_callbacks:
            cb(self)
        self.set_deployed()

    def change_model(self):
        self._success_label.text = ""
        self._success_label.hide()
        self._serve_button.show()
        self._device_select._select.enable()
        self._device_select.enable()
        self._change_model_button.hide()
        # @TODO: Ask web team to add message to list of request ready messages
        # Progress("model deployment canceled", 1).iter_done_report()
        Progress("Application is started ...", 1).iter_done_report()

    def _hide_info_after_change(self):
        self._model_full_info_card.collapse()
        self._model_full_info.hide()
        self._before_deploy_msg.show()

    def set_deployed(self, device: str = None):
        if device is not None:
            self._device_select.set_device(device)
        self._success_label.text = f"Model has been successfully loaded on {self._device_select.get_device().upper()} device"
        self._success_label.show()
        self._serve_button.hide()
        self._device_select._select.disable()
        self._device_select.disable()
        self._change_model_button.show()
        Progress("Model deployed", 1).iter_done_report()

    def show_deployed_model_info(self, inference):
        self.set_inference_settings(inference)
        self.set_project_meta(inference)
        self.set_model_info(inference)
        self._before_deploy_msg.hide()
        self._model_full_info.show()
        self._model_full_info_card.uncollapse()

    def set_inference_settings(self, inference):
        if len(inference.custom_inference_settings_dict.keys()) == 0:
            inference_settings_str = "# inference settings dict is empty"
        else:
            inference_settings_str = yaml.dump(inference.custom_inference_settings_dict)
        self._model_inference_settings_widget.set_text(inference_settings_str, "yaml")
        self._model_inference_settings_widget.show()

    def set_project_meta(self, inference):
        if self._get_classes_from_inference(inference) is None:
            logger.warn("Skip loading project meta.")
            self._model_classes_widget.hide()
            self._model_classes_plug.show()
            return

        self._model_classes_widget.set_project_meta(inference.model_meta)
        self._model_classes_plug.hide()
        self._model_classes_widget.show()

    def set_model_info(self, inference):
        info = inference.get_human_readable_info(replace_none_with="Not provided")
        self._model_info_widget.set_model_info(inference.task_id, info)

    def _get_classes_from_inference(self, inference) -> Optional[List[str]]:
        classes = None
        try:
            classes = inference.get_classes()
        except NotImplementedError:
            logger.warn(f"get_classes() function not implemented for {type(inference)} object.")
        except AttributeError:
            logger.warn("Probably, get_classes() function not working without model deploy.")
        except Exception as exc:
            logger.warn("Skip getting classes info due to exception")
            logger.exception(exc)

        if classes is None or len(classes) == 0:
            logger.warn(f"get_classes() function return {classes}; skip classes processing.")
            return None
        return classes

    def get_ui(self) -> Widgets.Widget:  # pylint: disable=method-hidden
        return Widgets.Container([self.serve_model_card])

    def add_content_to_default_ui(
        self, widgets: Union[Widgets.Widget, List[Widgets.Widget]]
    ) -> None:
        if isinstance(widgets, List):
            self._additional_ui_content.extend(widgets)
        else:
            self._additional_ui_content.append(widgets)

    def __add_content_and_model_info_to_default_ui(
        self,
        model_info_widget: Widgets.Widget,
    ) -> Callable:
        def decorator(get_ui):
            @wraps(get_ui)
            def wrapper(*args, **kwargs):
                ui = get_ui(*args, **kwargs)
                content = [ui, *self._additional_ui_content, model_info_widget]
                ui_with_info = Widgets.Container(content)
                return ui_with_info

            return wrapper

        return decorator(self.get_ui)
