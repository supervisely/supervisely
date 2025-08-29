from typing import List, Optional, Union, Tuple

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.app.widgets import NewExperiment, Dialog
from supervisely.nn.task_type import TaskType


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
        self.api = api
        self.project = project
        self.workspace_id = self.project.workspace_id
        self.team_id = self.project.team_id
        self.frameworks = frameworks
        self.widget = self._create_widget()

    def _create_widget(self) -> NewExperiment:
        """Creates the GUI widgets for the PretrainedModels node."""
        train_collections, val_collections = self._get_train_val_collections()
        split_mode = "collections" if train_collections and val_collections else "random"

        project_meta = ProjectMeta.from_json(self.api.project.get_meta(self.project.id))
        classes = [obj_cls.name for obj_cls in project_meta.obj_classes]

        content = NewExperiment(
            team_id=self.team_id,
            workspace_id=self.workspace_id,
            project_id=self.project.id,
            classes=classes,
            step=1,  # 5 - start with model selection
            filter_projects_by_workspace=True,
            project_types=[ProjectType.IMAGES],
            cv_task=None,  # TaskType.OBJECT_DETECTION,
            selected_frameworks=self.frameworks,
            train_val_split_mode=split_mode,  # only collections?
            train_collections=train_collections,
            val_collections=val_collections,
            # gui selectors disabled
            cv_task_selection_disabled=True,  # 1 - cv task selection
            project_selection_disabled=True,  # 2 - project selection
            classes_selection_disabled=False,  # 3 - classes selection
            train_val_split_selection_disabled=True,  # 4 - train/val split selection
            model_selection_disabled=False,  # 5 - model selection
            evaluation_selection_disabled=False,  # 9 - evaluation selection
            speed_test_selection_disabled=False,  # 9 - speed test selection
            framework_selection_disabled=self.frameworks is not None,
            architecture_selection_disabled=True,
        )

        @content.visible_changed
        def _on_visible_changed(visible: bool):
            print(f"NewExperiment visibility changed: {visible}")

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

    def _get_train_val_collections(self) -> Tuple[List[int], List[int]]:
        if self.project.type != ProjectType.IMAGES.value:
            return [], []
        train_collections, val_collections = [], []
        all_collections = self.api.entities_collection.get_list(self.project.id)
        for collection in all_collections:
            if collection.name == "All_train":
                train_collections.append(collection.id)
            elif collection.name == "All_val":
                val_collections.append(collection.id)
        return train_collections, val_collections
