from typing import List

from supervisely import Api, Project
from supervisely.app.widgets import Button, Card, Container, Text, TrainValSplits


class TrainValSplitsSelector:
    title = "Train / Val Splits"
    description = "Select train and val splits for training"
    lock_message = "Select input options to unlock"

    def __init__(self, api: Api, project_id: int, app_options: dict = {}):
        self.display_widgets = []
        self.api = api
        self.project_id = project_id

        # GUI Components
        self.train_val_splits = TrainValSplits(project_id)
        train_val_dataset_ids = {"train": [], "val": []}
        for _, dataset in api.dataset.tree(project_id):
            if dataset.name.lower() == "train" or dataset.name.lower() == "training":
                if dataset.items_count > 0:
                    train_val_dataset_ids["train"].append(dataset.id)
            elif dataset.name.lower() == "val" or dataset.name.lower() == "validation":
                if dataset.items_count > 0:
                    train_val_dataset_ids["val"].append(dataset.id)

        # Check nested dataset names
        train_count = len(train_val_dataset_ids["train"])
        val_count = len(train_val_dataset_ids["val"])
        if train_count > 0 and val_count > 0:
            self.train_val_splits.set_datasets_splits(
                train_val_dataset_ids["train"], train_val_dataset_ids["val"]
            )

        if train_count > 0 and val_count > 0:
            self.validator_text = Text("Train and val datasets are detected", status="info")
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
            collapsable=app_options.get("collapsable", False),
        )
        self.card.lock()

    @property
    def widgets_to_disable(self) -> list:
        return [self.train_val_splits]

    def validate_step(self) -> bool:
        split_method = self.train_val_splits.get_split_method()
        warning_text = "Using the same data for training and validation leads to overfitting, poor generalization and biased model selection."
        ensure_text = "Ensure this is intentional."

        if split_method == "Random":
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

        elif split_method == "Based on tags":
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

            # @TODO: handle button correctly if validation fails. Do not unlock next card until validation passes if returned False
            if tags_count[train_tag] == 0:
                self.validator_text.set(
                    text=f"Train tag '{train_tag}' is not present in any images. {ensure_text}",
                    status="error",
                )
            elif tags_count[val_tag] == 0:
                self.validator_text.set(
                    text=f"Val tag '{val_tag}' is not present in any images. {ensure_text}",
                    status="error",
                )

            elif train_tag == val_tag:
                self.validator_text.set(
                    text=f"Train and val tags are the same. {ensure_text} {warning_text}",
                    status="warning",
                )
            else:
                self.validator_text.set("Train and val tags are selected", status="success")

        elif split_method == "Based on datasets":
            train_dataset_id = self.get_train_dataset_ids()
            val_dataset_id = self.get_val_dataset_ids()

            # Check if datasets are not empty
            stats = self.api.project.get_stats(self.project_id)
            datasets_count = {}
            for dataset in stats["images"]["datasets"]:
                datasets_count[dataset["id"]] = {
                    "name": dataset["name"],
                    "total": dataset["imagesInDataset"],
                }

            empty_dataset_names = []
            for dataset_id in train_dataset_id + val_dataset_id:
                if datasets_count[dataset_id]["total"] == 0:
                    empty_dataset_names.append(datasets_count[dataset_id]["name"])

            if len(empty_dataset_names) > 0:
                if len(empty_dataset_names) == 1:
                    empty_ds_text = f"Selected dataset: {', '.join(empty_dataset_names)} is empty. {ensure_text}"
                else:
                    empty_ds_text = f"Selected datasets: {', '.join(empty_dataset_names)} are empty. {ensure_text}"

                self.validator_text.set(
                    text=empty_ds_text,
                    status="error",
                )

            elif train_dataset_id == val_dataset_id:
                self.validator_text.set(
                    text=f"Same datasets are selected for both train and val splits. {ensure_text} {warning_text}",
                    status="warning",
                )
            else:
                self.validator_text.set("Train and val datasets are selected", status="success")
        self.validator_text.show()
        return True

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
