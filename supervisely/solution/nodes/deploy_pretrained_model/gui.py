from collections import defaultdict

from supervisely.app.widgets import Container, Field, OneOf, RadioGroup
from supervisely.solution.components.deploy_model.gui import DeployModelGUI


class DeployPretrainedModelGUI(DeployModelGUI):

    def model_input_field(self) -> Field:
        if not hasattr(self, "_model_input_field"):
            frameworks_field = Field(
                title="Framework",
                description="Select one of the available frameworks.",
                content=self.framework_radio,
            )
            models_field = Field(
                title="Model",
                description="Select model to deploy.",
                content=self.models_one_of,
            )
            self._model_input_field = Container([frameworks_field, models_field])
        return self._model_input_field

    @property
    def models_one_of(self):
        if not hasattr(self, "_models_one_of"):
            self._models_one_of = OneOf(self.framework_radio)
        return self._models_one_of

    @property
    def framework_radio(self):
        if not hasattr(self, "_framework_radio"):
            framework_to_models = self._get_pretrained_models()
            self._framework_radio = RadioGroup(
                items=[
                    RadioGroup.Item(
                        value=fr,
                        label=f"{len(framework_to_models[fr])} models",
                        content=RadioGroup(
                            items=[
                                RadioGroup.Item(value=model_name)
                                for model_name in framework_to_models[fr]
                            ],
                            direction="vertical",
                            size="small",
                        ),
                    )
                    for fr in framework_to_models.keys()
                ],
                direction="vertical",
                size="small",
            )
        return self._framework_radio

    @property
    def model_name_input(self):
        if not hasattr(self, "_model_name_input"):

            def __get_model_name():
                selected_framework = self.framework_radio.get_value()
                selected_model = None
                for item in self.framework_radio._items:
                    if item.value == selected_framework:
                        selected_model = item.content.get_value()
                        break
                if selected_framework is not None and selected_model is not None:
                    return f"{selected_framework}/{selected_model}"
                return ""

            def __set_model_input_value(model: str):
                framework, model_name = model.split("/", 1)
                self.framework_radio.set_value(framework)
                for item in self.framework_radio._items:
                    if item.value == framework:
                        item.content.set_value(model_name)
                        break

            def __disable_model_input_field():
                self.framework_radio.disable()
                for item in self.framework_radio._items:
                    item.content.disable()

            def __enable_model_input_field():
                self.framework_radio.enable()
                for item in self.framework_radio._items:
                    item.content.enable()

            self._get_model_input_value = __get_model_name
            self._set_model_input_value = __set_model_input_value
            self._disable_model_input_field = __disable_model_input_field
            self._enable_model_input_field = __enable_model_input_field

            self._model_name_input = Container([self.framework_radio, self.models_one_of], gap=15)
        return self._model_name_input

    def _get_pretrained_models(self) -> list:
        """
        Returns a list of available pretrained models.
        This is a placeholder implementation and should be replaced with actual logic
        to fetch the list of pretrained models from a relevant source.
        """
        res = self._api.nn.ecosystem_models_api.list_models()
        framework_to_models = defaultdict(list)
        for model in res:
            framework = model.get("framework")
            name = model.get("name")
            if framework and name:
                framework_to_models[framework].append(name)

        return framework_to_models
