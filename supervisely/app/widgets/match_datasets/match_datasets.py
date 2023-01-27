from typing import List, Union
from supervisely import Api, Dataset
from supervisely.app import StateJson, DataJson
from supervisely import timeit
from supervisely.app.widgets import Widget


class MatchDatasets(Widget):
    """
    Compares Images by info.
    You can select fields to compare by. None to select all fields. Default - compare by hash
    Available field names:
    "id"
    "name"
    "link"
    "hash"
    "mime"
    "ext"
    "size"
    "width"
    "height"
    "labels_count"
    "dataset_id"
    "created_at"
    "updated_at"
    "meta"
    "path_original"
    "full_storage_url"
    "tags"
    """

    def __init__(
        self,
        left_datasets: List[Dataset],
        right_datasets: List[Dataset],
        left_name=None,
        right_name=None,
        compare_fields: Union[List[str], None] = ["hash"],
        widget_id=None,
    ):
        self._left_ds = left_datasets
        self._right_ds = right_datasets
        self._left_name = "Left Datasets" if left_name is None else left_name
        self._right_name = "Right Datasets" if right_name is None else right_name
        self._compare_fields = compare_fields
        self._api = Api()

        self._done = False
        self._loading = False

        self._image_dict = {"gt_images": {}, "pred_images": {}}
        self._results = None

        super().__init__(widget_id=widget_id, file_path=__file__)

        route_path = self.get_route_path("get_datasets_statistic")
        server = self._sly_app.get_server()
        server.post(route_path)(self.get_datasets_statistic)

    def get_json_data(self):
        return {"left_name": self._left_name, "right_name": self._right_name}

    def get_json_state(self):
        return {"done2": self._done, "loading2": self._loading}

    def process_items(self, ds_info1, collection1, ds_info2, collection2):
        ds_names = ds_info1.keys() | ds_info2.keys()

        results = []
        for idx, name in enumerate(ds_names):
            compare = {"dsIndex": idx}
            images1 = collection1.get(name, [])
            images2 = collection2.get(name, [])
            if len(images1) == 0:
                compare["message"] = ["unmatched (in GT project)"]
                compare["icon"] = [
                    ["zmdi zmdi-long-arrow-left", "zmdi zmdi-alert-circle-o"]
                ]

                compare["color"] = ["#F39C12"]
                compare["numbers"] = [-1]
                compare["left"] = {"name": ""}
                compare["right"] = {"name": name, "count": len(images2)}
            elif len(images2) == 0:
                compare["message"] = ["unmatched (in PRED project)"]
                compare["icon"] = [
                    ["zmdi zmdi-alert-circle-o", "zmdi zmdi-long-arrow-right"]
                ]
                compare["color"] = ["#F39C12"]
                compare["numbers"] = [-1]
                compare["left"] = {"name": name, "count": len(images1)}
                compare["right"] = {"name": ""}
            else:
                if self._compare_fields is None:
                    img_dict1 = {
                        img_info.name: img_info._asdict() for img_info in images1
                    }
                    img_dict2 = {
                        img_info.name: img_info._asdict() for img_info in images2
                    }
                else:
                    compare_fiels = set(self._compare_fields)
                    img_dict1 = {}
                    for img_info in images1:
                        img_dict1[img_info.name] = {
                            key: val
                            for key, val in img_info._asdict().items()
                            if key in compare_fiels
                        }
                    img_dict2 = {}
                    for img_info in images2:
                        img_dict2[img_info.name] = {
                            key: val
                            for key, val in img_info._asdict().items()
                            if key in compare_fiels
                        }

                matched = []
                diff = []  # same names but different hashes or image sizes
                same_names = img_dict1.keys() & img_dict2.keys()
                for img_name in same_names:
                    dest = (
                        matched if img_dict1[img_name] == img_dict2[img_name] else diff
                    )
                    dest.append([img_dict1[img_name], img_dict2[img_name]])  # extend

                uniq1 = [img_dict1[name] for name in img_dict1.keys() - same_names]
                uniq2 = [img_dict2[name] for name in img_dict2.keys() - same_names]

                compare["message"] = [
                    "matched",
                    "conflicts",
                    "unique (left)",
                    "unique (right)",
                ]
                compare["icon"] = [
                    ["zmdi zmdi-check"],
                    ["zmdi zmdi-close"],
                    ["zmdi zmdi-plus-circle-o"],
                    ["zmdi zmdi-plus-circle-o"],
                ]
                compare["color"] = ["green", "red", "#20a0ff", "#20a0ff"]
                compare["numbers"] = [len(matched), len(diff), len(uniq1), len(uniq2)]
                compare["left"] = {"name": name, "count": len(images1)}
                compare["right"] = {"name": name, "count": len(images2)}

            results.append(compare)

        self._results = results
        return results

    def _get_all_images(self, datasets):
        ds_info = {}
        ds_images = {}
        ws_to_team = {}
        for dataset in datasets:
            ds_info[dataset.name] = dataset
            images = self._api.image.get_list(dataset.id)
            ds_images[dataset.name] = images
        return (
            ds_info,
            ds_images,
        )

    @timeit
    def get_datasets_statistic(self):
        StateJson()[self.widget_id]["loading2"] = True
        StateJson().send_changes()

        ds_info1, ds_images1 = self._get_all_images(self._left_ds)
        ds_info2, ds_images2 = self._get_all_images(self._right_ds)
        result = self.process_items(ds_info1, ds_images1, ds_info2, ds_images2)
        intersected_keys = list(set(list(ds_images1)) & set(list(ds_images2)))

        for intersected_key in intersected_keys:
            self._image_dict["gt_images"][intersected_key] = []
            self._image_dict["pred_images"][intersected_key] = []

            for gt_element in ds_images1[intersected_key]:
                for pred_element in ds_images2[intersected_key]:
                    if (
                        gt_element.hash == pred_element.hash
                        and gt_element.name == pred_element.name
                    ):
                        self._image_dict["gt_images"][intersected_key].append(
                            gt_element
                        )
                        self._image_dict["pred_images"][intersected_key].append(
                            pred_element
                        )

        DataJson()[self.widget_id]["table"] = result
        DataJson().send_changes()
        StateJson()[self.widget_id]["done2"] = True
        StateJson()[self.widget_id]["loading2"] = False
        StateJson().send_changes()
