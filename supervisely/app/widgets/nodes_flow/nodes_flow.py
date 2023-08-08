import copy
from typing import List, Optional
from supervisely.app.widgets import Widget
from supervisely.app.content import StateJson

from supervisely.app.widgets.nodes_flow.option_components import (
    OptionComponent,
    WidgetOptionComponent,
    ButtonOptionComponent,
    CheckboxOptionComponent,
    InputOptionComponent,
    IntegerOptionComponent,
    NumberOptionComponent,
    SelectOptionComponent,
    SliderOptionComponent,
    TextOptionComponent
)


class NodesFlow(Widget):

    class OptionComponent(OptionComponent):
        pass

    class WidgetOptionComponent(WidgetOptionComponent):
        pass

    class ButtonOptionComponent(ButtonOptionComponent):
        pass

    class CheckboxOptionComponent(CheckboxOptionComponent):
        pass

    class InputOptionComponent(InputOptionComponent):
        pass

    class IntegerOptionComponent(IntegerOptionComponent):
        pass

    class NumberOptionComponent(NumberOptionComponent):
        pass

    class SelectOptionComponent(SelectOptionComponent):
        pass

    class SliderOptionComponent(SliderOptionComponent):
        pass

    class TextOptionComponent(TextOptionComponent):
        pass

    class Node:
        class Input:
            def __init__(self, name, label: Optional[str] = None):
                self.name = name
                self.label = label

            def to_json(self):
                j = {"name": self.name}
                if self.label is not None:
                    j["options"] = {"displayName": self.label}
                return j

        class Output(Input):
            pass

        class Option:
            def __init__(self, name: str, option_component: OptionComponent):
                self.name = name
                self.option_component = option_component

            def to_json(self):
                return {"name": self.name, **self.option_component.to_json()}

        def __init__(
            self,
            id,
            name="Node",
            width: Optional[int] = 200,
            options: List[Option] = [],
            inputs: List[Input] = [],
            outputs: List[Output] = [],
        ):
            self.id = id
            self.name = name
            self._width = width
            self.options = options
            self.inputs = inputs
            self.outputs = outputs

        def to_json(self):
            return {
                "id": self.id,
                "name": self.name,
                "width": self._width,
                "options": [option.to_json() for option in self.options],
                "inputs": [i.to_json() for i in self.inputs],
                "outputs": [o.to_json() for o in self.outputs],
            }

    def __init__(
        self,
        nodes: List[Node] = [],
        height: str = None,
        widget_id: str = None,
    ):
        self.nodes = nodes
        self.height = height if height is not None else "500px"
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "height": self.height,
        }

    def get_json_state(self):
        return {
            "flowState": {},
            "flow": {"nodes": [node.to_json() for node in self.nodes], "edges": []},
        }

    def add_node(self, node: Node):
        self.nodes.append(node)
        StateJson()[self.widget_id]["flow"]["nodes"].append(node.to_json())
        StateJson().send_changes()

    def pop_node(self, idx: int):
        try:
            removed = StateJson()[self.widget_id]["flow"]["nodes"].pop(idx)
            StateJson().send_changes()
        except IndexError:
            return None
        return removed

    def delete_node_by_id(self, id: str):
        for i, node in enumerate(self.nodes):
            if node.id == id:
                return self.pop_node(i)
        return None

    def get_nodes_json(self):
        return copy.deepcopy(StateJson()[self.widget_id]["flow"]["nodes"])
    
    def get_edges_json(self):
        return copy.deepcopy(StateJson()[self.widget_id]["flow"]["edges"])

    def get_nodes_state_json(self):
        return copy.deepcopy(StateJson()[self.widget_id]["flowState"])
