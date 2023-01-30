from supervisely.app.widgets import Widget
from supervisely.app import DataJson, StateJson
from supervisely import (
    TagMetaCollection,
    ObjClassCollection,
    TagMeta,
    ObjClass,
    TagValueType,
    color,
)
from typing import Union

class MatchTagMetasOrClasses(Widget):
    def __init__(
        self,
        left_collection: Union[TagMetaCollection, ObjClassCollection],
        right_collection: Union[TagMetaCollection, ObjClassCollection],
        selectable: bool = False,
        widget_id: str = None,
    ):
        self._left_collection = left_collection
        self._right_collection = right_collection
        self._selectable = selectable

        self._table = self._get_table()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {"table": self._table, "selectable": self._selectable}

    def get_json_state(self):
        return {"selected": []}
    
    def set(
        self,
        left_collection: Union[TagMetaCollection, ObjClassCollection, None],
        right_collection: Union[TagMetaCollection, ObjClassCollection, None],      
    ):
        if left_collection is not None:
            self._left_collection = left_collection
        if right_collection is not None:
            self._right_collection = right_collection
        self._table = self._get_table()
        DataJson()[self.widget_id]["table"] = self._table
        DataJson().send_changes()
        StateJson()[self.widget_id] = self.get_json_state()
        StateJson().send_changes()

    def get_stat(self):
        stat = {
            "match": [],
            "only_left": [],
            "only_right": [],
            "different_shape": [],
            "different_value_type": [],
            "different_one_of_options": [],
        }
        for row in self._table:
            message = row.get("infoMessage")
            name = row["name1"] if "name1" in row.keys() else row["name2"]
            if message == "Match":
                stat["match"].append(name)
            elif message == "Not found in right Project":
                stat["only_left"].append(name)
            elif message == "Not found in left Project":
                stat["only_right"].append(name)
            elif message == "Different shape":
                stat["different_shape"].append(name)
            elif message == "Different value type":
                stat["different_value_type"].append(name)
            elif message == "Type OneOf: conflict of possible values":
                stat["different_one_of_options"].append(name)
        return stat

    def _get_table(self):
        items1 = {item.name: 1 for item in self._left_collection}
        items2 = {item.name: 1 for item in self._right_collection}
        names = items1.keys() | items2.keys()
        mutual = items1.keys() & items2.keys()
        diff1 = items1.keys() - mutual
        diff2 = items2.keys() - mutual

        match = []
        differ = []
        missed = []

        def set_info(d, index, meta):
            d[f"name{index}"] = meta.name
            d[f"color{index}"] = color.rgb2hex(meta.color)
            if type(meta) is ObjClass:
                d[f"shape{index}"] = meta.geometry_type.geometry_name()
                d[f"shapeIcon{index}"] = "zmdi zmdi-shape"
            else:
                meta: TagMeta
                d[f"shape{index}"] = meta.value_type
                d[f"shapeIcon{index}"] = "zmdi zmdi-label"

        for name in names:
            compare = {}
            meta1 = self._left_collection.get(name)
            if meta1 is not None:
                set_info(compare, 1, meta1)
            meta2 = self._right_collection.get(name)
            if meta2 is not None:
                set_info(compare, 2, meta2)

            if name in mutual:
                flag = True
                if (
                    type(meta1) is ObjClass
                    and meta1.geometry_type != meta2.geometry_type
                ):
                    flag = False
                    diff_msg = "Different shape"
                if type(meta1) is TagMeta:
                    meta1: TagMeta
                    meta2: TagMeta
                    if meta1.value_type != meta2.value_type:
                        flag = False
                        diff_msg = "Different value type"
                    if meta1.value_type == TagValueType.ONEOF_STRING and set(meta1.possible_values) != set(meta2.possible_values):
                        flag = False
                        diff_msg = "Type OneOf: conflict of possible values"

                if flag is False:
                    compare["infoMessage"] = diff_msg
                    compare["infoColor"] = "red"
                    compare["infoIcon"] = (["zmdi zmdi-flag"],)
                    differ.append(compare)
                else:
                    compare["infoMessage"] = "Match"
                    compare["infoColor"] = "green"
                    compare["infoIcon"] = (["zmdi zmdi-check"],)
                    match.append(compare)
            else:
                if name in diff1:
                    compare["infoMessage"] = "Not found in right Project"
                    compare["infoIcon"] = [
                        "zmdi zmdi-alert-circle-o",
                        "zmdi zmdi-long-arrow-right",
                    ]
                    compare["iconPosition"] = "right"
                else:
                    compare["infoMessage"] = "Not found in left Project"
                    compare["infoIcon"] = [
                        "zmdi zmdi-long-arrow-left",
                        "zmdi zmdi-alert-circle-o",
                    ]
                compare["infoColor"] = "#FFBF00"
                missed.append(compare)

        table = []
        if match:
            match.sort(key=lambda x: x["name1"])
        table.extend(match)
        if differ:
            differ.sort(key=lambda x: x["name1"])
        table.extend(differ)
        table.extend(missed)

        return table

    def get_selected(self):
        return StateJson()[self.widget_id]["selected"]
