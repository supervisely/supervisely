from typing import Union, List
from supervisely.app import DataJson
from supervisely.app.widgets import (
    Widget,
    TagMetaView,
    Checkbox,
    Grid,
    generate_id,
    Button,
)
from supervisely.annotation.tag_meta import TagMeta
from supervisely.annotation.tag_meta_collection import TagMetaCollection


class TagMetasList(Widget):
    def __init__(
        self,
        tag_metas: Union[TagMetaCollection, List[TagMeta]],
        show_type_text: bool = True,
        limit_long_names: bool = False,
        selectable: bool = False,
        columns: int = 1,  # 1 means vertical layout
        widget_id: str = None,
    ):
        self._tag_metas = tag_metas
        self._selectable = selectable
        self._columns = columns

        if type(tag_metas) is list:
            self._tag_metas = TagMetaCollection(self._tag_metas)

        self._name_to_tag = {}
        self._name_to_view = {}
        self._name_to_checkbox = {}
        self._select_all_btn = None
        self._deselect_all_btn = None

        for tag_meta in self._tag_metas:
            self._name_to_tag[tag_meta.name] = tag_meta
            self._name_to_view[tag_meta.name] = TagMetaView(
                tag_meta, show_type_text, limit_long_names, widget_id=generate_id()
            )
            grid_items = list(self._name_to_view.values())
            if self._selectable is True:
                self._select_all_btn = Button(
                    "Select all",
                    button_type="text",
                    show_loading=False,
                    icon="zmdi zmdi-check-all",
                    widget_id=generate_id(),
                )
                self._deselect_all_btn = Button(
                    "Deselect all",
                    button_type="text",
                    show_loading=False,
                    icon="zmdi zmdi-square-o",
                    widget_id=generate_id(),
                )
                current_checkbox = Checkbox(
                    self._name_to_view[tag_meta.name],
                    checked=True,
                    widget_id=generate_id(),
                )
                self._name_to_checkbox[tag_meta.name] = current_checkbox
                grid_items = list(self._name_to_checkbox.values())

                @self._select_all_btn.click
                def select_all():
                    for k, v in self._name_to_checkbox.items():
                        v: Checkbox
                        v.check()
                    DataJson()[self.widget_id]["num_selected"] = len(self._name_to_checkbox)
                    DataJson().send_changes()

                @self._deselect_all_btn.click
                def deselect_all():
                    for k, v in self._name_to_checkbox.items():
                        v: Checkbox
                        v.uncheck()
                    DataJson()[self.widget_id]["num_selected"] = 0
                    DataJson().send_changes()

                @current_checkbox.value_changed
                def checkbox_changed(is_checked):
                    DataJson()[self.widget_id]["num_selected"] = len(self.get_selected_tag_names())
                    DataJson().send_changes()

        self._content = Grid(
            widgets=grid_items,
            columns=self._columns,
            widget_id=generate_id(),
        )
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        res = {"selectable": self._selectable}
        if self._selectable is True:
            res["num_selected"] = len(self.get_selected_tag_names())
        return res

    def get_selected_tag_names(self):
        if self._selectable is False:
            raise ValueError(
                "Class selection is disabled, because TagMetasList widget was created with argument 'selectable == False'"
            )
        results = []
        for k, v in self._name_to_checkbox.items():
            v: Checkbox
            if v.is_checked():
                results.append(k)
        return results

    def get_json_state(self):
        return None
