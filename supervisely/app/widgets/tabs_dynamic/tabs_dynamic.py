from io import StringIO
from pathlib import Path
from typing import List, Optional, Dict

from supervisely.app import StateJson
from supervisely.app.widgets import Widget, Editor
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

class MyYAML(YAML):
    def dump(self, data, stream=None, **kw):
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        YAML.dump(self, data, stream, **kw)
        if inefficient:
            return stream.getvalue()

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
        if Path(filepath_or_raw_yaml[-50:]).is_file():
            data_source = open(filepath_or_raw_yaml, "r")
        else:
            data_source = filepath_or_raw_yaml

        yaml = MyYAML()
        self._data = yaml.load(data_source)
        common_data = self._data.copy()
        
        self._items_dict = {'common': None}
        self._items = []
        for label, yaml_fragment in self._data.items():
            if isinstance(yaml_fragment, CommentedMap):
                yaml_str = yaml.dump(yaml_fragment)
                editor = Editor(yaml_str, language_mode='yaml', height_px=250)
                self._items_dict[label] = editor
                self._items.append(TabsDynamic.TabPane(label=label, content=editor))
                del common_data[label]

        yaml_str = yaml.dump(common_data)
        editor = Editor(yaml_str, language_mode='yaml', height_px=250)
        self._items_dict['common'] = editor
        self._items.insert(0, TabsDynamic.TabPane(label='common', content=editor))

        assert len(set(self._items_dict.keys())) == len(self._items_dict.keys()), ValueError("All of tab labels should be unique.")
        assert len(self._items_dict.keys()) == len(self._items), ValueError("labels length must be equal to contents length in Tabs widget.")

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