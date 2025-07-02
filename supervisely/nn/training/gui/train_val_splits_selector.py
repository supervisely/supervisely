from typing import List

from supervisely import Api, Project
from supervisely.app.widgets import Button, Card, Container, Text, TrainValSplits
from supervisely.api.module_api import ApiField

class TrainValSplitsSelector:
    title = "Train / Val Splits"
    description = "Select train and val splits for training"
    lock_message = "Select previous step to unlock"

    def __init__(self, api: Api, project_id: int, app_options: dict = {}):
        # Init widgets
        self.train_val_splits = None
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        self.display_widgets = []
        self.app_options = app_options
        self.api = api
        self.project_id = project_id

        # GUI Components
        split_methods = self.app_options.get("train_val_split_methods", [])
        if len(split_methods) == 0:
            split_methods = ["Random", "Based on tags", "Based on datasets", "Based on collections"]
        random_split = "Random" in split_methods
        tag_split = "Based on tags" in split_methods
        ds_split = "Based on datasets" in split_methods
        coll_split = "Based on collections" in split_methods

        self.train_val_splits = TrainValSplits(
            project_id, None, random_split, tag_split, ds_split, collections_splits=coll_split
        )

        # check for collections with "train" and "val" prefixes
        all_collections = self.api.entities_collection.get_list(self.project_id)
        train_collections = []
        val_collections = []
        collections_found = False
        for collection in all_collections:
            if collection.name.lower().startswith("train"):
                train_collections.append(collection.id)
            elif collection.name.lower().startswith("val"):
                val_collections.append(collection.id)
        
        if len(train_collections) > 0 and len(val_collections) > 0:
            self.train_val_splits.set_collections_splits(train_collections, val_collections)
            self.validator_text = Text(
                "Train and val collections are detected", status="info"
            )
            self.validator_text.show()
            collections_found = True
        else:
            self.validator_text = Text("")
            self.validator_text.hide()
        

        def _extend_with_nested(root_ds):
            nested = self.api.dataset.get_nested(self.project_id, root_ds.id)
            nested_ids = [ds.id for ds in nested]
            return [root_ds.id] + nested_ids

        if not collections_found:
            train_val_dataset_ids = {"train": set(), "val": set()}
            for _, dataset in self.api.dataset.tree(self.project_id):
                ds_name = dataset.name.lower()

                if ds_name in {"train", "training"}:
                    for _id in _extend_with_nested(dataset):
                        train_val_dataset_ids["train"].add(_id)

                elif ds_name in {"val", "validation", "test", "testing"}:
                    for _id in _extend_with_nested(dataset):
                        train_val_dataset_ids["val"].add(_id)

            train_val_dataset_ids["train"] = list(train_val_dataset_ids["train"])
            train_val_dataset_ids["val"] = list(train_val_dataset_ids["val"])

            train_count = len(train_val_dataset_ids["train"])
            val_count = len(train_val_dataset_ids["val"])

            if train_count > 0 and val_count > 0:
                self.train_val_splits.set_datasets_splits(
                    train_val_dataset_ids["train"], train_val_dataset_ids["val"]
                )

            if train_count > 0 and val_count > 0:
                if train_count == val_count == 1:
                    self.validator_text = Text("train and val datasets are detected", status="info")
                else:
                    self.validator_text = Text(
                        "Multiple train and val datasets are detected. Check manually if selection is correct",
                        status="info",
                    )
                self.validator_text.show()
            else:
                self.validator_text = Text("")
                self.validator_text.hide()

        self.button = Button("Select")
        self.display_widgets.extend([self.train_val_splits, self.validator_text, self.button])
        # -------------------------------- #

        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
            collapsable=self.app_options.get("collapsable", False),
        )
        self.card.lock()

    @property
    def widgets_to_disable(self) -> list:
        return [self.train_val_splits]

    def validate_step(self) -> bool:
        split_method = self.train_val_splits.get_split_method()
        warning_text = "Using the same data for training and validation leads to overfitting, poor generalization and biased model selection."
        ensure_text = "Ensure this is intentional."
        is_valid = False

        def validate_random_split():
            train_ratio = self.train_val_splits.get_train_split_percent()
            val_ratio = self.train_val_splits.get_val_split_percent()

            # Define common warning text
            ensure_text_random_split = (
                "Consider reallocating to ensure efficient learning and validation."
            )

            # Validate splits
            if train_ratio == val_ratio:
                self.validator_text.set(
                    text="Train and validation splits are equal (50:50). This is inefficient for standard training. "
                    f"{ensure_text}",
                    status="warning",
                )
            elif train_ratio > 90:
                self.validator_text.set(
                    text="Training split exceeds 90%. This may leave insufficient data for validation. Ensure you have enough data for validation.",
                    status="warning",
                )
            elif val_ratio > train_ratio:
                self.validator_text.set(
                    text=f"Validation split is larger than the training split. {ensure_text_random_split}",
                    status="warning",
                )
            elif train_ratio < 70:
                self.validator_text.set(
                    text="Training split is below 70%. This may limit the model's learning capability. "
                    f"{ensure_text_random_split}",
                    status="warning",
                )
            else:
                self.validator_text.set(
                    text="Train and validation splits are selected.",
                    status="success",
                )
            return True

        def validate_based_on_tags():
            train_tag = self.train_val_splits.get_train_tag()
            val_tag = self.train_val_splits.get_val_tag()

            # Check if tags are present on any item in the project
            stats = self.api.project.get_stats(self.project_id)
            tags_count = {}
            for item in stats["imageTags"]["items"]:
                tag_name = item["tagMeta"]["name"]
                tag_total = item["total"]
                tags_count[tag_name] = tag_total

            for object_tags in stats["objectTags"]["items"]:
                tag_name = object_tags["tagMeta"]["name"]
                tag_total = object_tags["total"]
                if tag_name in tags_count:
                    tags_count[tag_name] += tag_total
                else:
                    tags_count[tag_name] = tag_total

            if tags_count[train_tag] == 0:
                self.validator_text.set(
                    text=f"Train tag '{train_tag}' is not present in any images. {ensure_text}",
                    status="error",
                )
                return False
            elif tags_count[val_tag] == 0:
                self.validator_text.set(
                    text=f"Val tag '{val_tag}' is not present in any images. {ensure_text}",
                    status="error",
                )
                return False
            elif train_tag == val_tag:
                self.validator_text.set(
                    text=f"Train and val tags are the same. {ensure_text} {warning_text}",
                    status="warning",
                )
                return True
            else:
                self.validator_text.set("Train and val tags are selected", status="success")
                return True

        def validate_based_on_datasets():
            train_dataset_id = self.get_train_dataset_ids()
            val_dataset_id = self.get_val_dataset_ids()
            if train_dataset_id is None and val_dataset_id is None:
                self.validator_text.set("No datasets are selected", status="error")
                return False

            if train_dataset_id is None:
                self.validator_text.set("No train dataset is selected", status="error")
                return False

            if val_dataset_id is None:
                self.validator_text.set("No val dataset is selected", status="error")
                return False

            # Check if datasets are not empty
            filters = [{ ApiField.FIELD: ApiField.ID, ApiField.OPERATOR: "in", ApiField.VALUE: train_dataset_id + val_dataset_id}]
            selected_datasets = self.api.dataset.get_list(self.project_id, filters, recursive=True)
            datasets_count = {}
            for dataset in selected_datasets:
                datasets_count[dataset.id] = {
                    "name": dataset.name,
                    "total": dataset.images_count,
                }

            empty_dataset_names = []
            for dataset_id in train_dataset_id + val_dataset_id:
                if datasets_count[dataset_id]["total"] == 0:
                    empty_dataset_names.append(datasets_count[dataset_id]["name"])

            if len(empty_dataset_names) > 0:
                if len(empty_dataset_names) == len(train_dataset_id + val_dataset_id):
                    empty_ds_text = f"All selected datasets are empty. {ensure_text}"
                    self.validator_text.set(
                        text=empty_ds_text,
                        status="error",
                    )
                    return False

                if len(empty_dataset_names) == 1:
                    empty_ds_text = f"Selected dataset: {', '.join(empty_dataset_names)} is empty. {ensure_text}"
                else:
                    empty_ds_text = f"Selected datasets: {', '.join(empty_dataset_names)} are empty. {ensure_text}"

                self.validator_text.set(
                    text=empty_ds_text,
                    status="error",
                )
                return True

            elif train_dataset_id == val_dataset_id:
                self.validator_text.set(
                    text=f"Same datasets are selected for both train and val splits. {ensure_text} {warning_text}",
                    status="warning",
                )
                return True
            else:
                self.validator_text.set("Train and val datasets are selected", status="success")
                return True

        def validate_based_on_collections():
            train_collection_id = self.train_val_splits.get_train_collections_ids()
            val_collection_id = self.train_val_splits.get_val_collections_ids()
            if train_collection_id is None and val_collection_id is None:
                self.validator_text.set("No collections are selected", status="error")
                return False
            if len(train_collection_id) == 0 or len(val_collection_id) == 0:
                self.validator_text.set("Collections are not selected", status="error")
                return False
            if set(train_collection_id) == set(val_collection_id):
                self.validator_text.set(
                    text=f"Same collections are selected for both train and val splits. {ensure_text} {warning_text}",
                    status="warning",
                )
                return True
            from supervisely.api.entities_collection_api import CollectionTypeFilter

            train_items = set()
            empty_train_collections = []
            for collection_id in train_collection_id:
                items = self.api.entities_collection.get_items(
                    collection_id=collection_id,
                    project_id=self.project_id,
                    collection_type=CollectionTypeFilter.DEFAULT,
                )
                train_items.update([item.id for item in items])
                if len(items) == 0:
                    empty_train_collections.append(collection_id)
            val_items = set()
            empty_val_collections = []
            for collection_id in val_collection_id:
                items = self.api.entities_collection.get_items(
                    collection_id=collection_id,
                    project_id=self.project_id,
                    collection_type=CollectionTypeFilter.DEFAULT,
                )
                val_items.update([item.id for item in items])
                if len(items) == 0:
                    empty_val_collections.append(collection_id)
            if len(train_items) == 0 and len(val_items) == 0:
                self.validator_text.set(
                    text="All selected collections are empty. ",
                    status="error",
                )
                return False
            if len(empty_train_collections) > 0 or len(empty_val_collections) > 0:
                empty_collections_text = "Selected collections are empty. "
                if len(empty_train_collections) > 0:
                    empty_collections_text += f"train: {', '.join(empty_train_collections)}. "
                if len(empty_val_collections) > 0:
                    empty_collections_text += f"val: {', '.join(empty_val_collections)}. "
                empty_collections_text += f"{ensure_text}"
                self.validator_text.set(
                    text=empty_collections_text,
                    status="error",
                )
                return True

            else:
                self.validator_text.set("Train and val collections are selected", status="success")
                return True

        if split_method == "Random":
            is_valid = validate_random_split()

        elif split_method == "Based on tags":
            is_valid = validate_based_on_tags()

        elif split_method == "Based on datasets":
            is_valid = validate_based_on_datasets()
        elif split_method == "Based on collections":
            is_valid = validate_based_on_collections()

        # @TODO: handle button correctly if validation fails. Do not unlock next card until validation passes if returned False
        self.validator_text.show()
        return is_valid

    def set_sly_project(self, project: Project) -> None:
        self.train_val_splits._project_fs = project

    def get_split_method(self) -> str:
        return self.train_val_splits.get_split_method()

    def get_train_dataset_ids(self) -> List[int]:
        return self.train_val_splits._train_ds_select.get_selected_ids()

    def set_train_dataset_ids(self, dataset_ids: List[int]) -> None:
        self.train_val_splits._train_ds_select.set_selected_ids(dataset_ids)

    def get_val_dataset_ids(self) -> List[int]:
        return self.train_val_splits._val_ds_select.get_selected_ids()

    def set_val_dataset_ids(self, dataset_ids: List[int]) -> None:
        self.train_val_splits._val_ds_select.set_selected_ids(dataset_ids)

    def get_train_collection_ids(self) -> List[int]:
        return self.train_val_splits._train_collections_select.get_selected_ids()

    def set_train_collection_ids(self, collection_ids: List[int]) -> None:
        self.train_val_splits._train_collections_select.set_selected_ids(collection_ids)

    def get_val_collection_ids(self) -> List[int]:
        return self.train_val_splits._val_collections_select.get_selected_ids()

    def set_val_collection_ids(self, collection_ids: List[int]) -> None:
        self.train_val_splits._val_collections_select.set_selected_ids(collection_ids)
