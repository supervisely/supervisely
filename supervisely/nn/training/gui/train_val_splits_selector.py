from typing import List, Union

from supervisely import Api
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    NotificationBox,
    Text,
    TrainValSplits,
)


class TrainValSplitsSelector:
    title = "Train/Val Splits Selector"

    def __init__(self, api: Api, project_id: int):
        self.train_val_splits = TrainValSplits(project_id)

        # Check Train / Val datasets to set default
        dataset_paths = []
        dataset_names = []
        for parents, dataset in api.dataset.tree(project_id):
            dataset_name = "/".join(parents + [dataset.name])
            dataset_paths.append(dataset_name)
            dataset_names.append(dataset.name)

        # Check duplicate dataset names
        train_count = dataset_names.count("train")
        val_count = dataset_names.count("val")
        self.notification_box = NotificationBox(
            title="Multiple 'train' and 'val' datasets detected",
            description="Project consists of multiple datasets named 'train' or 'val', this includes nested datasets. Please select the correct datasets manually.",
            box_type="warning",
        )
        self.notification_box.hide()

        # if train_count > 1 or val_count > 1:
        # set default tab by datasets
        # self.notification_box.show()
        # else:
        # set default tab by datasets
        # self.notification_box.hide()

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        container = Container(
            [
                self.train_val_splits,
                self.notification_box,  # may be not needed
                self.validator_text,
                self.button,
            ]
        )
        self.card = Card(
            title="Train / Val Splits",
            description="Select train and val splits for training",
            content=container,
            lock_message="Select input options to unlock",
        )
        self.card.lock()

    @property
    def widgets_to_disable(self):
        return [self.train_val_splits]

    def validate_step(self):
        # @TODO: add validation
        # split_method = self.train_val_splits.get_split_method()
        # if split_method == "random":
        #     train_percent = self.train_val_splits.get_train_percent()
        # self.validator_text.hide()
        # train_dataset_id = self.get_train_dataset_id()
        # val_dataset_id = self.get_val_dataset_id()

        # error_messages = []
        # if train_dataset_id is None:
        #     error_messages.append("Train dataset is not selected")
        # if val_dataset_id is None:
        #     error_messages.append("Val dataset is not selected")

        # if error_messages:
        #     self.validator_text.set(text="; ".join(error_messages), status="error")
        #     self.validator_text.show()
        #     return False

        # if train_dataset_id == val_dataset_id:
        #     self.validator_text.set(
        #         text=(
        #             "Train and val datasets are the same. "
        #             "Using the same dataset for training and validation leads to overfitting, "
        #             "poor generalization, biased model selection, "
        #             "misleading metrics, and reduced robustness on unseen data."
        #         ),
        #         status="warning",
        #     )
        #     self.validator_text.show()
        #     return True
        # else:
        #     self.validator_text.set(text="Train and val datasets are selected", status="success")
        #     self.validator_text.show()
        #     return True
        return True

    def get_split_method(self):
        return self.train_val_splits.get_split_method()

    def get_train_dataset_ids(self):
        return self.train_val_splits._train_ds_select.get_selected_ids()

    def set_train_dataset_ids(self, dataset_ids: List[int]):
        self.train_val_splits._train_ds_select.set_selected_ids(dataset_ids)

    def get_val_dataset_ids(self):
        return self.train_val_splits._val_ds_select.get_selected_ids()

    def set_val_dataset_ids(self, dataset_ids: List[int]):
        self.train_val_splits._val_ds_select.set_selected_ids(dataset_ids)
