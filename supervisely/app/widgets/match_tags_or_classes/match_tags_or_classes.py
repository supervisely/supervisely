from supervisely.app.widgets import Widget
from supervisely.app import DataJson
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
        widget_id: str = None,
    ):
        self._left_collection = left_collection
        self._right_collection = right_collection

        self._table = self._get_table()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {"table": self._table}

    def get_json_state(self):
        return {}
    
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

    def _get_table(
        self,
        diff_msg="Automatic conversion to rectangle format",
    ):
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

            compare["infoMessage"] = "Match"
            compare["infoColor"] = "green"
            if name in mutual:
                flag = True
                if (
                    type(meta1) is ObjClass
                    and meta1.geometry_type != meta2.geometry_type
                ):
                    flag = False
                if type(meta1) is TagMeta:
                    meta1: TagMeta
                    meta2: TagMeta
                    if meta1.value_type != meta2.value_type:
                        flag = False
                    if meta1.value_type == TagValueType.ONEOF_STRING:
                        if set(meta1.possible_values) != set(meta2.possible_values):
                            diff_msg = "Type OneOf: conflict of possible values"
                        flag = False

                if flag is False:
                    compare["infoMessage"] = diff_msg
                    compare["infoColor"] = "red"
                    compare["infoIcon"] = (["zmdi zmdi-flag"],)
                    differ.append(compare)
                else:
                    compare["infoIcon"] = (["zmdi zmdi-check"],)
                    match.append(compare)
            else:
                if name in diff1:
                    compare["infoMessage"] = "Not found in PRED Project"
                    compare["infoIcon"] = [
                        "zmdi zmdi-alert-circle-o",
                        "zmdi zmdi-long-arrow-right",
                    ]
                    compare["iconPosition"] = "right"
                else:
                    compare["infoMessage"] = "Not found in GT Project"
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
