from typing import Any, List, Optional
from supervisely.app.widgets import Widget


class OptionComponent:
    def __init__(
        self,
        component_name: str,
        default_value: Optional[Any] = None,
        sidebar_component: Optional[str] = None,
        options: Optional[dict] = {},
    ):
        self.component_name = component_name
        self.default_value = default_value
        self.sidebar_component = sidebar_component
        self.options = options

    def to_json(self):
        return {
            "component": self.component_name,
            "value": self.default_value,
            "sidebarComponent": self.sidebar_component,
            "options": self.options,
        }


class HtmlOptionComponent(OptionComponent):
    def __init__(
        self,
        html: str,
        sidebar_component: Optional[OptionComponent] = None,
        sidebar_width: Optional[int] = None,
    ):
        options = {}
        sidebar_component_name = None
        if sidebar_component is not None:
            sidebar_component_name = sidebar_component.component_name
            sidebar_template = sidebar_component.options["template"]
            options["sidebarTemplate"] = f"<div>{sidebar_template}</div>"
            options["sidebarWidth"] = sidebar_width
        options["template"] = f"<div>{html}</div>"
        super().__init__(
            component_name="SlyFlowOptionRenderer",
            options=options,
            sidebar_component=sidebar_component_name
        )


class WidgetOptionComponent(HtmlOptionComponent):
    def __init__(self, widget: Widget, sidebar_component=None, sidebar_width: Optional[int] = None):
        super().__init__(
            widget.to_html(), sidebar_component=sidebar_component, sidebar_width=sidebar_width
        )


class ButtonOptionComponent(OptionComponent):
    """A button that opens the sidebar when clicked. The label of the button is determined by the name of the option."""

    def __init__(
        self,
        sidebar_component: Optional[OptionComponent] = None,
        sidebar_width: Optional[int] = None,
    ):
        sidebar_component_name = None
        options = {}
        if sidebar_component is not None:
            sidebar_component_name = sidebar_component.component_name
            options = sidebar_component.options
            options["sidebarWidth"] = sidebar_width
        super().__init__(
            component_name="ButtonOption", sidebar_component=sidebar_component_name, options=options
        )


class CheckboxOptionComponent(OptionComponent):
    """A checkbox for setting boolean values."""

    def __init__(self, default_value: bool = False):
        super().__init__(component_name="CheckboxOption", default_value=default_value)


class InputOptionComponent(OptionComponent):
    """A simple text field. The option name will be displayed as placeholder."""

    def __init__(self, default_value: str = ""):
        super().__init__(component_name="InputOption", default_value=default_value)


class IntegerOptionComponent(OptionComponent):
    """A numeric up/down field for integer values."""

    def __init__(
        self,
        min: Optional[int] = None,
        max: Optional[int] = None,
        default_value: Optional[int] = 0,
    ):
        options = {}
        if min is not None:
            options["min"] = min
        if max is not None:
            options["max"] = max
        super().__init__(
            component_name="IntegerOption",
            default_value=default_value,
            options=options,
        )


class NumberOptionComponent(OptionComponent):
    """A numeric up/down field for numeric values."""

    def __init__(
        self,
        min: Optional[float] = None,
        max: Optional[float] = None,
        default_value: Optional[float] = 0,
    ):
        options = {}
        if min is not None:
            options["min"] = min
        if max is not None:
            options["max"] = max
        super().__init__(
            component_name="NumberOption",
            default_value=default_value,
            options=options,
        )


class SelectOptionComponent(OptionComponent):
    """A dropdown select which allows the user to choose one of predefined values."""

    class Item:
        def __init__(self, value: str, label: Optional[str] = None):
            self.value = value
            self.label = value if label is None else label

        def to_json(self):
            return {"text": self.label, "value": self.value}

    def __init__(self, items: List[Item], default_value: Optional[str] = None):
        options = {"items": [item.to_json() for item in items]}
        super().__init__(
            component_name="SelectOption",
            default_value=default_value,
            options=options,
        )


class SliderOptionComponent(OptionComponent):
    """A slider for choosing a value in a range. The user can also click on the slider to set a specific value."""

    def __init__(
        self,
        min: float,
        max: float,
        default_value: Optional[float] = None,
    ):
        options = {
            "min": min,
            "max": max,
        }
        default_value = min if default_value is None else default_value
        super().__init__(
            component_name="SliderOption",
            default_value=default_value,
            options=options,
        )


class TextOptionComponent(OptionComponent):
    """Displays arbitrary strings"""

    def __init__(self, text: str):
        super().__init__(component_name="TextOption", default_value=text)


class SidebarNodeInfoOptionComponent(OptionComponent):
    def __init__(self, sidebar_template: str, sidebar_width: Optional[int] = None):
        options = {
            "sidebarTemplate": f"<div>{sidebar_template}</div>", 
            "sidebarWidth": sidebar_width
        }
        super().__init__(
            component_name="SlyFlowOptionRenderer",
            sidebar_component="SlyFlowOptionRenderer",
            options=options,
        )
