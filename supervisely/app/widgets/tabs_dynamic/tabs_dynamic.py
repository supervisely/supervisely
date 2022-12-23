from typing import List, Optional, Dict
from supervisely.app import StateJson
from supervisely.app.widgets import Widget, Editor
import yaml
from pathlib import Path

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal



class TabsDynamic(Widget):
    class TabPane:
        def __init__(self, label: str, content: Widget):
            self.label = label
            self.name = label  # identifier corresponding to the active tab
            self.content = content

    def __init__(
        self,
        filepath_or_raw_yaml: str,
        type: Optional[Literal["card", "border-card"]] = "border-card",
        widget_id=None,
    ):  
        if Path(filepath_or_raw_yaml).is_file():
            data_source = open(filepath_or_raw_yaml, "r")
        else:
            data_source = filepath_or_raw_yaml

        try:
            self._data = yaml.safe_load(data_source)
        except yaml.YAMLError as exc:
            print(exc)
        
        
        self._items = []
        self._items_dict = {'common': {}}
        for key, val in self._data.items():
            if isinstance(val, dict):
                self._items_dict[key] = val
            else:
                self._items_dict['common'][key] = val
        
        for label, data in self._items_dict.items():
            self._items.append(TabsDynamic.TabPane(label=label, content=Editor(yaml.dump(data), language_mode='yaml', height_px=250)))
        
        assert len(set(self._items_dict.keys())) == len(self._items_dict.keys()), ValueError("All of tab labels should be unique.")
        assert len(self._items_dict.keys()) == len(self._items), ValueError("labels length must be equal to contents length in Tabs widget.")
        # assert len(labels) <= 10, ValueError("You can specify up to 10 tabs.")

        self._value = list(self._items_dict.keys())[0]
        self._type = type
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {"type": self._type}

    def get_json_state(self) -> Dict:
        return {"value": self._value}

    def set_active_tab(self, value: str):
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_active_tab(self) -> str:
        return StateJson()[self.widget_id]["value"]
