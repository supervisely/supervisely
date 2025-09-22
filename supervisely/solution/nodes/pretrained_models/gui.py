from typing import List, Optional, Tuple, Union

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import Dialog, NewExperiment
from supervisely.nn.task_type import TaskType
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.solution.utils import find_agent, get_last_split_collection


class PretrainedModelsGUI:
    """
    GUI components for the PretrainedModels node.
    """

    def __init__(
        self,
        api: Api,
        project: ProjectInfo,
        frameworks: Optional[List[str]] = None,
        widget_id: Optional[str] = None,
    ):
        self._api = api
        self.project = project
        self.workspace_id = self.project.workspace_id
        self.team_id = self.project.team_id
        self.frameworks = frameworks
        self.widget = self._create_widget()

    def _create_widget(self) -> NewExperiment:
        """Creates the GUI widgets for the PretrainedModels node."""
        train_datasets, val_datasets = self._get_train_val_datasets()
        train_collections, val_collections = self._get_train_val_collections()

        if train_collections and val_collections:
            split_mode = "collections"
            train_datasets, val_datasets = None, None
        elif train_datasets and val_datasets:
            split_mode = "datasets"
            train_collections, val_collections = None, None
        else:
            split_mode = "random"
            train_collections, val_collections = None, None
            train_datasets, val_datasets = None, None

        disable_split = False  # ! TODO: split_mode != "random"
        project_meta = ProjectMeta.from_json(self._api.project.get_meta(self.project.id))
        classes = [obj_cls.name for obj_cls in project_meta.obj_classes]
        agent_id = self._find_agent()

        content = NewExperiment(
            team_id=self.team_id,
            workspace_id=self.workspace_id,
            project_id=self.project.id,
            classes=classes,
            step=2,  # - start with model selection
            filter_projects_by_workspace=True,
            project_types=[ProjectType.IMAGES],
            cv_task=TaskType.OBJECT_DETECTION,
            selected_frameworks=self.frameworks,
            train_val_split_mode=split_mode,
            train_collections=train_collections,
            val_collections=val_collections,
            training_datasets=train_datasets,
            val_datasets=val_datasets,
            agent_id=agent_id,
            run_evaluation=True,
            # gui selectors disabled
            cv_task_selection_disabled=True,  # 1 - cv task selection
            project_selection_disabled=True,  # 2 - project selection
            classes_selection_disabled=False,  # 3 - classes selection
            train_val_split_selection_disabled=disable_split,  # 4 - train/val split selection
            model_selection_disabled=False,  # 5 - model selection
            evaluation_selection_disabled=False,  # 9 - evaluation selection
            speed_test_selection_disabled=False,  # 9 - speed test selection
            framework_selection_disabled=self.frameworks is not None,
            architecture_selection_disabled=True,
        )

        @content.visible_changed
        def _on_visible_changed(visible: bool):
            if visible:
                train_collections, val_collections = self._get_train_val_collections()
                content.train_collections = train_collections
                content.val_collections = val_collections

        return content

    @property
    def modal(self):
        """
        Create the modal dialog for automation settings.
        """
        if not hasattr(self, "_modal"):
            self._modal = Dialog(
                title="Pretrained Models",
                content=self.widget,
                size="tiny",
            )
        return self._modal

    def _get_train_val_datasets(self) -> Tuple[List[int], List[int]]:
        if self.project.type != ProjectType.IMAGES.value:
            return [], []
        train_datasets, val_datasets = [], []
        all_datasets = self._api.dataset.get_list(self.project.id)
        for dataset in all_datasets:
            if "train" in dataset.name:
                train_datasets.append(dataset.id)
            elif "val" in dataset.name:
                val_datasets.append(dataset.id)
        return train_datasets, val_datasets

    def _get_train_val_collections(self) -> Tuple[List[int], List[int]]:
        last_train, _ = get_last_split_collection(self._api, self.project.id, "train_")
        last_val, _ = get_last_split_collection(self._api, self.project.id, "val_")
        if last_train and last_val:
            return [last_train.id], [last_val.id]
        return [], []

    def _find_agent(self):
        return find_agent(self._api, self.team_id)
