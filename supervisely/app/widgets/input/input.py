from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class Input(Widget):
    def __init__(
        self,
        value: str = "",
        type: int = "text",  # "text"|"textarea"
        minlength: int = 0,
        maxlength: int = 1000,
        placeholder: str = "",
        size: str = "small",  # "large"|"small"|"mini" if type != "textarea"
        icon: str = None,
        rows: int = None,  # if type == "textarea"
        autosize: bool = False,  # if type == "textarea"
        auto_complete: str = "off",  # "on"|"off"
        name: str = None,  # native html input attribute
        readonly: bool = False,  # native html input attribute
        max: int = None,  # native html input attribute
        min: int = None,  # native html input attribute
        step: int = None,  # native html input attribute
        resize: str = "none",  # "none"|"both"|"horizontal"|"vertical"
        autofocus: bool = False,  # native html input attribute
        form: str = None,
        widget_id: str = None
    ):
        self._value = value
        self._type = type
        self._minlength = minlength
        self._maxlength = maxlength
        self._placeholder = placeholder
        if icon is None:
            self._icon = ""
        else:
            self._icon = f'<i class="{icon}" style="margin-right: 5px"></i>'

        self._resize = resize

        if type != "textarea":
            self._size = size
        else:
            self._size = None

        if type == "textarea":
            self._rows = rows
            self._autosize = autosize
        else:
            self._rows = None
            self._autosize = False

        # original input attributes
        self._auto_complete = auto_complete
        self._name = name
        self._readonly = readonly
        self._max = max
        self._min = min
        self._step = step
        self._autofocus = autofocus
        self._form = form

        self._widget_id = widget_id

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "type": self._type,
            "minlength": self._minlength,
            "maxlength": self._maxlength,
            "placeholder": self._placeholder,
            "icon": self._icon,
            "resize": self._resize,
            "size": self._size,
            "rows": self._rows,
            "autosize": self._autosize,
            "auto_complete": self._auto_complete,
            "name": self._name,
            "readonly": self._readonly,
            "max": self._max,
            "min": self._min,
            "step": self._step,
            "autofocus": self._autofocus,
            "form": self._form
        }

    def get_json_state(self):
        return {"value": self._value}

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value

    @property
    def minlength(self):
        return self._min

    @minlength.setter
    def minlength(self, value):
        self._minlength = value
        DataJson()[self.widget_id]["minlength"] = self._minlength

    @property
    def maxlength(self):
        return self._max

    @maxlength.setter
    def maxlength(self, value):
        self._maxlength = value
        DataJson()[self.widget_id]["maxlength"] = self._maxlength

    @property
    def readonly(self):
        return self._readonly

    @readonly.setter
    def readonly(self, value: bool):
        self._readonly = value
        DataJson()[self.widget_id]["readonly"] = self._readonly

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value: str):
        if value not in ["mini", "small", "large"] and self._type == "textarea":
            return
        self._size = value
        DataJson()[self.widget_id]["readonly"] = self._size