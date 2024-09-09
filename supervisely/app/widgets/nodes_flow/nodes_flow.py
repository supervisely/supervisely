from __future__ import annotations
import copy
from typing import List, Literal, Optional
from supervisely.app.widgets import Widget
from supervisely.app.content import StateJson

from supervisely.app.widgets.nodes_flow.option_components import (
    OptionComponent,
    HtmlOptionComponent,
    WidgetOptionComponent,
    ButtonOptionComponent,
    CheckboxOptionComponent,
    InputOptionComponent,
    IntegerOptionComponent,
    NumberOptionComponent,
    SelectOptionComponent,
    SliderOptionComponent,
    TextOptionComponent,
    SidebarNodeInfoOptionComponent,
)


class NodesFlow(Widget):
    class OptionComponent(OptionComponent):
        pass

    class HtmlOptionComponent(HtmlOptionComponent):
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

    class SidebarNodeInfoOptionComponent(SidebarNodeInfoOptionComponent):
        pass

    class Node:
        class Input:
            def __init__(self, name, label: Optional[str] = None, color: Optional[str] = None):
                self.name = name
                self.label = label
                self.color = color

            def to_json(self):
                j = {"name": self.name}
                if self.label is not None:
                    j.setdefault("options", {})["displayName"] = self.label
                if self.color is not None:
                    j.setdefault("options", {})["type"] = self.color
                return j

        class Output(Input):
            pass

        class Option:
            def __init__(self, name: str, option_component: OptionComponent):
                if isinstance(option_component, NodesFlow.SidebarNodeInfoOptionComponent):
                    name = "sidebarNodeInfo"
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
            inputs_up: bool = False,
            position: Optional[dict] = None,
            header_color: Optional[str] = None,
            header_text_color: Optional[str] = None,
            icon: Optional[str] = None,
            icon_background_color: Optional[str] = None,
        ):
            self.id = id
            self.name = name
            self._width = width
            self._inputs_up = inputs_up
            self.options = options
            self.inputs = inputs
            self.outputs = outputs
            self._position = position
            self._header_background_color = header_color
            self._header_color = header_text_color
            self._icon = icon
            self._icon_background_color = icon_background_color

        def to_json(self):
            return {
                "id": self.id,
                "name": self.name,
                "width": self._width,
                "twoColumn": self._inputs_up,
                "options": [option.to_json() for option in self.options],
                "inputs": [i.to_json() for i in self.inputs],
                "outputs": [o.to_json() for o in self.outputs],
                "position": self._position,
                "header": {
                    "backgroundColor": self._header_background_color,
                    "color": self._header_color,
                    "icon": {
                        "backgroundColor": self._icon_background_color,
                        "class": self._icon,
                    },
                },
            }

        def set_position(self, position):
            self._position = position

    class Routes:
        SAVE = "save_cb"
        FLOW_CHANGED = "flow_changed_cb"
        FLOW_STATE_CHANGED = "flow_state_changed_cb"
        CONTEXT_MENU_CLICKED = "context_menu_item_click_cb"
        SIDEBAR_TOGGLED = "sidebar_toggled_cb"
        ITEM_DROPPED = "item_dropped_cb"
        NODE_REMOVED = "node_removed_cb"

    def __init__(
        self,
        nodes: List[Node] = [],
        height: str = None,
        context_menu: dict = None,
        color_theme: Literal["light", "dark"] = "light",
        drag_and_drop_menu: dict = None,
        drag_and_drop_menu_width: int = "400px",
        show_save: bool = True,
        widget_id: str = None,
    ):
        self._nodes = nodes
        self._height = height if height is not None else "500px"
        self._context_menu = context_menu
        self._color_theme = color_theme
        self._dd_menu = drag_and_drop_menu
        self._show_dd_area = False
        self._dd_section_width = drag_and_drop_menu_width
        self._show_save = show_save
        if self._dd_menu:
            self._show_dd_area = True

        self._save_handled = False
        self._flow_change_handled = False
        self._flow_state_change_handled = False
        self._contex_menu_item_click_handled = False
        self._sidebar_toggle_handled = False
        self._item_dropped_handled = False
        self._node_removed_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "height": self._height,
            "contextMenuItems": self._context_menu,
            "colorTheme": self._color_theme,
            "nodeTypeList": self._dd_menu,
            "showDDArea": self._show_dd_area,
            "ddSectionWidth": self._dd_section_width,
            "interfaceTypes": [{"name": f"{color}", "color": color} for color in []],
            "showSave": self._show_save,
        }

    def get_json_state(self):
        return {
            "flowState": {},
            "flow": {"nodes": [node.to_json() for node in self._nodes], "edges": []},
        }

    def add_node(self, node: Node):
        self._nodes.append(node)
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
        for i, node in enumerate(self._nodes):
            if node.id == id:
                return self.pop_node(i)
        return None

    def get_flow_json(self):
        return copy.deepcopy(StateJson()[self.widget_id]["flow"])

    def get_nodes_json(self):
        return copy.deepcopy(StateJson()[self.widget_id]["flow"]["nodes"])

    def get_edges_json(self):
        return copy.deepcopy(StateJson()[self.widget_id]["flow"]["edges"])

    def get_flow_state(self):
        return copy.deepcopy(StateJson()[self.widget_id]["flowState"])

    def get_nodes_state_json(self):
        """Alias for get_flow_state"""
        return self.get_flow_state()

    def on_save(self, func):
        route_path = self.get_route_path(NodesFlow.Routes.SAVE)
        server = self._sly_app.get_server()
        self._save_handled = True

        @server.post(route_path)
        def _click():
            func()

        return _click

    def flow_changed(self, func):
        route_path = self.get_route_path(NodesFlow.Routes.FLOW_CHANGED)
        server = self._sly_app.get_server()
        self._flow_change_handled = True

        @server.post(route_path)
        def _click():
            func()

        return _click

    def flow_state_changed(self, func):
        route_path = self.get_route_path(NodesFlow.Routes.FLOW_STATE_CHANGED)
        server = self._sly_app.get_server()
        self._flow_state_change_handled = True

        @server.post(route_path)
        def _click():
            func()

        return _click

    def clear(self):
        StateJson()[self.widget_id]["flow"]["nodes"] = []
        StateJson()[self.widget_id]["flow"]["edges"] = []
        StateJson()[self.widget_id]["flowState"] = {}

        self._nodes = []
        StateJson().send_changes()

    def set_edges(self, edges: List[dict]):
        StateJson()[self.widget_id]["flow"]["edges"] = edges
        StateJson().send_changes()

    def context_menu_clicked(self, func):
        route_path = self.get_route_path(NodesFlow.Routes.CONTEXT_MENU_CLICKED)
        server = self._sly_app.get_server()
        self._contex_menu_item_click_handled = True

        @server.post(route_path)
        def _click():
            item = StateJson()[self.widget_id]["selectedContextItem"]
            func(item)

        return _click

    def sidebar_toggled(self, func):
        route_path = self.get_route_path(NodesFlow.Routes.SIDEBAR_TOGGLED)
        server = self._sly_app.get_server()
        self._sidebar_toggle_handled = True

        @server.post(route_path)
        def _click():
            func()

        return _click

    def item_dropped(self, func):
        route_path = self.get_route_path(NodesFlow.Routes.ITEM_DROPPED)
        server = self._sly_app.get_server()
        self._item_dropped_handled = True

        @server.post(route_path)
        def _click():
            item = StateJson()[self.widget_id]["droppedItem"]
            func(item)

        return _click

    def node_removed(self, func):
        route_path = self.get_route_path(NodesFlow.Routes.NODE_REMOVED)
        server = self._sly_app.get_server()
        self._node_removed_handled = True

        @server.post(route_path)
        def _click():
            item = StateJson()[self.widget_id]["removedNode"]
            func(item)

        return _click
