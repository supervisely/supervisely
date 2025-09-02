from typing import List, Literal, Optional, Union

from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.app import StateJson
from supervisely.app.widgets import Widget
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType


class NewExperiment(Widget):
    class SplitMode:
        DATASETS = "datasets"
        COLLECTIONS = "collections"
        RANDOM = "random"

    class ExportType:
        PYTORCH = "pytorch"
        ONNX = "onnx"
        TENSORRT = "tensorrt"

    class Routes:
        VISIBLE_CHANGED = "visible_changed_cb"
        APP_STARTED = "app_started_cb"

    def __init__(
        self,
        team_id: Optional[int] = None,
        workspace_id: Optional[int] = None,
        user_id: Optional[int] = None,
        project_id: Optional[int] = None,
        redirect_to_session: bool = False,
        filter_projects_by_workspace: bool = False,
        project_types: Optional[List[ProjectType]] = None,
        cv_task: Optional[str] = None,
        classes: Optional[List[str]] = None,
        model_id: Optional[str] = None,
        agent_id: Optional[int] = None,
        export: List[Literal["pytorch", "onnx", "tensorrt"]] = [ExportType.PYTORCH],
        run_evaluation: bool = True,
        run_speed_test: bool = False,
        experiment_name: Optional[str] = None,
        step: Optional[str] = "1",
        train_val_split_mode: Literal["datasets", "collections", "random"] = SplitMode.RANDOM,
        training_datasets: Optional[Union[List[int], List[str]]] = None,
        val_datasets: Optional[Union[List[int], List[str]]] = None,
        train_collections: Optional[List[int]] = None,
        val_collections: Optional[List[int]] = None,
        random_train_percentage: int = 80,
        selected_frameworks: Optional[List[str]] = None,
        selected_architectures: Optional[List[str]] = None,
        cv_task_selection_disabled: bool = False,
        project_selection_disabled: bool = False,
        classes_selection_disabled: bool = False,
        model_selection_disabled: bool = False,
        evaluation_selection_disabled: bool = False,
        speed_test_selection_disabled: bool = False,
        train_val_split_selection_disabled: bool = False,
        framework_selection_disabled: bool = False,
        architecture_selection_disabled: bool = False,
        widget_id: Optional[str] = None,
    ):
        self._api = Api()
        self._user_id = user_id
        if self._user_id is None:
            self._user_id = self._api.user.get_my_info().id

        self._workspace_id = workspace_id
        self._team_id = team_id
        if self._team_id is None and self._workspace_id is not None:
            workspace = self._api.workspace.get_info_by_id(self._workspace_id)
            if workspace is None:
                raise ValueError(f"Workspace with ID {self._workspace_id} not found.")
            self._team_id = workspace.team_id

        # Options and form data:
        self._redirect_to_session = redirect_to_session
        self._filter_projects_by_workspace = filter_projects_by_workspace if workspace_id else False
        if not isinstance(project_types, list):
            raise TypeError("project_types must be a list of ProjectType.")
        self._project_types = [type.value for type in project_types]
        self._cv_task = cv_task
        self._project_id = project_id
        if classes is not None and all(isinstance(name, str) for name in classes):
            meta = ProjectMeta.from_json(self._api.project.get_meta(project_id))
            classes = [obj_cls.sly_id for obj_cls in meta.obj_classes if obj_cls.name in classes]
        if classes is None:
            classes = []
        self._classes = classes
        self._agent_id = agent_id
        self._export = export
        self._run_evaluation = run_evaluation
        self._run_speed_test = run_speed_test
        self._experiment_name = experiment_name
        self._step = step
        self._model_id = model_id
        self._selected_frameworks = selected_frameworks
        self._selected_architectures = selected_architectures

        # Train/Val split settings:
        self._training_datasets = self._validate_datasets(training_datasets)
        self._val_datasets = self._validate_datasets(val_datasets)
        self._train_collections = self._validate_collections(train_collections)
        self._val_collections = self._validate_collections(val_collections)
        self._random_train_percentage = random_train_percentage or 80
        self._train_val_split_mode = self._validate_split_mode(train_val_split_mode)

        # Widget selectors disabled by default:
        self._cv_task_selection_disabled = cv_task_selection_disabled if cv_task else False
        self._project_selection_disabled = project_selection_disabled if project_id else False
        self._classes_selection_disabled = classes_selection_disabled if classes else False
        self._train_val_split_selection_disabled = train_val_split_selection_disabled
        self._model_selection_disabled = model_selection_disabled if model_id else False
        self._evaluation_disabled = evaluation_selection_disabled if run_evaluation else False
        self._speed_test_disabled = speed_test_selection_disabled if run_speed_test else False
        self._framework_selection_disabled = framework_selection_disabled
        self._architecture_selection_disabled = architecture_selection_disabled

        self._visible_handled = False
        self._app_started_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _validate_split_mode(self, mode: Optional[str] = None) -> str:
        if mode not in [
            NewExperiment.SplitMode.DATASETS,
            NewExperiment.SplitMode.COLLECTIONS,
            NewExperiment.SplitMode.RANDOM,
            None,
        ]:
            raise ValueError(
                f"Invalid train_val_split_mode: {mode}. "
                f"Must be one of {NewExperiment.SplitMode.DATASETS}, "
                f"{NewExperiment.SplitMode.COLLECTIONS}, or {NewExperiment.SplitMode.RANDOM}."
            )
        if mode is None:
            if self._training_datasets or self._val_datasets:
                mode = NewExperiment.SplitMode.DATASETS
            elif self._train_collections or self._val_collections:
                mode = NewExperiment.SplitMode.COLLECTIONS
            else:
                mode = NewExperiment.SplitMode.RANDOM
        return mode

    def _validate_collections(self, collections: Union[List[int], List[str], None]) -> List[int]:
        if collections is None:
            return []
        if not isinstance(collections, list):
            collections = [collections]
        if all(isinstance(item, str) for item in collections):
            filters = [
                {
                    ApiField.FIELD: ApiField.NAME,
                    ApiField.OPERATOR: "in",
                    ApiField.VALUE: collections,
                }
            ]
            if collections == []:
                filters = None
            collections = self._api.entities_collection.get_list(
                project_id=self._project_id, filters=filters
            )
            return [collection.id for collection in collections]
        elif all(isinstance(item, int) for item in collections):
            return collections
        else:
            raise TypeError("collections must be a list of integers or strings.")

    def _validate_datasets(self, datasets: Union[List[int], List[str], None]) -> List[int]:
        if datasets is None:
            return []
        if not isinstance(datasets, list):
            datasets = [datasets]
        if all(isinstance(item, str) for item in datasets):
            filters = [
                {
                    ApiField.FIELD: ApiField.NAME,
                    ApiField.OPERATOR: "in",
                    ApiField.VALUE: datasets,
                }
            ]
            datasets = self._api.dataset.get_list(parent_id=self._project_id, filters=filters)
            return [dataset.id for dataset in datasets]
        elif all(isinstance(item, int) for item in datasets):
            return datasets
        else:
            raise TypeError("datasets must be a list of integers or strings.")

    def get_json_data(self):
        return {
            "userId": self._user_id,
            "teamId": self._team_id,
            "workspaceId": self._workspace_id,
            "options": {
                "redirectToSession": self._redirect_to_session,
                "filterProjectsByWorkspace": self._filter_projects_by_workspace,
                "projectTypes": self._project_types,
                "cvTaskSelectionDisabled": self._cv_task_selection_disabled,
                "projectSelectionDisabled": self._project_selection_disabled,
                "classesSelectionDisabled": self._classes_selection_disabled,
                "modelSelectionDisabled": self._model_selection_disabled,
                "trainValSplitSelectionDisabled": self._train_val_split_selection_disabled,
                "evaluationSelectionDisabled": self._evaluation_disabled,
                "speedTestSelectionDisabled": self._speed_test_disabled,
                "frameworkSelectionDisabled": self._framework_selection_disabled,
                "architectureSelectionDisabled": self._architecture_selection_disabled,
                "selectedFrameworks": self._selected_frameworks,
                "selectedArchitectures": self._selected_architectures,
                "allowEmptyExperimentName": True,
                # @TODO: remove this before the branch is merged
                "version": "solutions-train-test",
                "isBranch": True,
            },
        }

    def get_json_state(self):
        return {
            "visible": False,
            "appId": None,
            "modelId": None,
            "taskId": None,
            "step": self._step,
            "form": {
                "cvTask": self._cv_task,
                "selectedProjectId": self._project_id,
                "selectedClasses": self._classes,
                "trainValSplit": {
                    "mode": self._train_val_split_mode,  # "datasets", "collections", "random"
                    "randomTrainPercentage": self._random_train_percentage,
                    "trainDatasets": self._training_datasets,
                    "valDatasets": self._val_datasets,
                    "trainCollections": self._train_collections,
                    "valCollections": self._val_collections,
                },
                "selectedModelId": self._model_id,  # "911-fr3-YOLOv8 test"
                "nodeId": self._agent_id,
                "export": {
                    "pytorch": self._export and "pytorch" in self._export,
                    "onnx": self._export and "onnx" in self._export,
                    "tensorrt": self._export and "tensorrt" in self._export,
                },
                "runEvaluation": self._run_evaluation,
                "runSpeedTest": self._run_speed_test,
                "experimentName": self._experiment_name,
            },
        }

    @property
    def visible(self) -> bool:
        return StateJson()[self.widget_id]["visible"]

    @visible.setter
    def visible(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("Visible must be a boolean value.")
        StateJson()[self.widget_id]["visible"] = value
        StateJson().send_changes()

    @property
    def step(self) -> Optional[str]:
        self._step = StateJson()[self.widget_id]["step"]
        return self._step

    @step.setter
    def step(self, value: Optional[str]):
        self._step = value
        StateJson()[self.widget_id]["step"] = value
        StateJson().send_changes()

    @property
    def cv_task(self) -> Optional[str]:
        self._cv_task = StateJson()[self.widget_id]["form"]["cvTask"]
        return self._cv_task

    @cv_task.setter
    def cv_task(self, value: Optional[str]):
        if value is not None and not isinstance(value, str):
            raise TypeError("cv_task must be a string or None.")
        self._cv_task = value
        StateJson()[self.widget_id]["form"]["cvTask"] = value
        StateJson().send_changes()

    @property
    def project_id(self) -> Optional[int]:
        self._project_id = StateJson()[self.widget_id]["form"]["selectedProjectId"]
        return self._project_id

    @project_id.setter
    def project_id(self, value: Optional[int]):
        if value is not None and not isinstance(value, int):
            raise TypeError("project_id must be an integer or None.")
        self._project_id = value
        StateJson()[self.widget_id]["form"]["selectedProjectId"] = value
        StateJson().send_changes()

    @property
    def classes(self) -> Optional[List[int]]:
        self._classes = StateJson()[self.widget_id]["form"]["selectedClasses"]
        return self._classes

    @classes.setter
    def classes(self, value: Optional[List[Union[str, int]]]):
        if value is not None and not isinstance(value, list):
            raise TypeError("classes must be a list or None.")
        if value is not None and all(isinstance(name, str) for name in value):
            meta = ProjectMeta.from_json(self._api.project.get_meta_by_id(self._project_id))
            value = [obj_cls.sly_id for obj_cls in meta.obj_classes if obj_cls.name in value]
        self._classes = value
        StateJson()[self.widget_id]["form"]["selectedClasses"] = value
        StateJson().send_changes()

    @property
    def model_id(self) -> Optional[str]:
        self._model_id = StateJson()[self.widget_id]["form"]["selectedModelId"]
        return self._model_id

    @model_id.setter
    def model_id(self, value: Optional[str]):
        if value is not None and not isinstance(value, str):
            raise TypeError("model_id must be a string or None.")
        self._model_id = value
        StateJson()[self.widget_id]["form"]["selectedModelId"] = value
        StateJson().send_changes()

    @property
    def agent_id(self) -> Optional[int]:
        self._agent_id = StateJson()[self.widget_id]["form"]["nodeId"]
        return self._agent_id

    @agent_id.setter
    def agent_id(self, value: Optional[int]):
        if value is not None and not isinstance(value, int):
            raise TypeError("agent_id must be an integer or None.")
        self._agent_id = value
        StateJson()[self.widget_id]["form"]["nodeId"] = value
        StateJson().send_changes()

    @property
    def export(self) -> List[Literal["pytorch", "onnx", "tensorrt"]]:
        export_dict = StateJson()[self.widget_id]["form"]["export"]
        export_list = []
        if export_dict.get(NewExperiment.ExportType.PYTORCH, False):
            export_list.append(NewExperiment.ExportType.PYTORCH)
        if export_dict.get(NewExperiment.ExportType.ONNX, False):
            export_list.append(NewExperiment.ExportType.ONNX)
        if export_dict.get(NewExperiment.ExportType.TENSORRT, False):
            export_list.append(NewExperiment.ExportType.TENSORRT)
        self._export = export_list
        return export_list

    @export.setter
    def export(self, value: List[Literal["pytorch", "onnx", "tensorrt"]]):
        if not isinstance(value, list):
            raise TypeError("export must be a list.")
        if not all(isinstance(item, str) for item in value):
            raise TypeError("All items in export must be strings.")
        self._export = value
        StateJson()[self.widget_id]["form"]["export"] = {
            "pytorch": "pytorch" in value,
            "onnx": "onnx" in value,
            "tensorrt": "tensorrt" in value,
        }
        StateJson().send_changes()

    @property
    def run_evaluation(self) -> bool:
        self._run_evaluation = StateJson()[self.widget_id]["form"]["runEvaluation"]
        return self._run_evaluation

    @run_evaluation.setter
    def run_evaluation(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("run_evaluation must be a boolean value.")
        self._run_evaluation = value
        StateJson()[self.widget_id]["form"]["runEvaluation"] = value
        StateJson().send_changes()

    @property
    def run_speed_test(self) -> bool:
        self._run_speed_test = StateJson()[self.widget_id]["form"]["runSpeedTest"]
        return self._run_speed_test

    @run_speed_test.setter
    def run_speed_test(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("run_speed_test must be a boolean value.")
        self._run_speed_test = value
        StateJson()[self.widget_id]["form"]["runSpeedTest"] = value
        StateJson().send_changes()

    @property
    def experiment_name(self) -> Optional[str]:
        self._experiment_name = StateJson()[self.widget_id]["form"]["experimentName"]
        return self._experiment_name

    @experiment_name.setter
    def experiment_name(self, value: Optional[str]):
        if value is not None and not isinstance(value, str):
            raise TypeError("experiment_name must be a string or None.")
        self._experiment_name = value
        StateJson()[self.widget_id]["form"]["experimentName"] = value
        StateJson().send_changes()

    @property
    def training_datasets(self) -> List[int]:
        self._training_datasets = StateJson()[self.widget_id]["form"]["trainValSplit"][
            "trainDatasets"
        ]
        return self._training_datasets

    @training_datasets.setter
    def training_datasets(self, value: Optional[Union[List[int], List[str]]]):
        if not isinstance(value, list):
            raise TypeError("training_datasets must be a list of dataset IDs or names.")
        self._training_datasets = self._validate_datasets(value)
        StateJson()[self.widget_id]["form"]["trainValSplit"][
            "trainDatasets"
        ] = self._training_datasets
        StateJson().send_changes()

    @property
    def val_datasets(self) -> List[int]:
        self._val_datasets = StateJson()[self.widget_id]["form"]["trainValSplit"]["valDatasets"]
        return self._val_datasets

    @val_datasets.setter
    def val_datasets(self, value: Optional[Union[List[int], List[str]]]):
        if not isinstance(value, list):
            raise TypeError("val_datasets must be a list of dataset IDs or names.")
        self._val_datasets = self._validate_datasets(value)
        StateJson()[self.widget_id]["form"]["trainValSplit"]["valDatasets"] = self._val_datasets
        StateJson().send_changes()

    @property
    def train_collections(self) -> List[int]:
        self._train_collections = StateJson()[self.widget_id]["form"]["trainValSplit"][
            "trainCollections"
        ]
        return self._train_collections

    @train_collections.setter
    def train_collections(self, value: Optional[Union[List[int], List[str]]]):
        if not isinstance(value, list):
            raise TypeError("train_collections must be a list of collection IDs or names.")
        self._train_collections = self._validate_collections(value)
        StateJson()[self.widget_id]["form"]["trainValSplit"][
            "trainCollections"
        ] = self._train_collections
        StateJson().send_changes()

    @property
    def val_collections(self) -> List[int]:
        self._val_collections = StateJson()[self.widget_id]["form"]["trainValSplit"][
            "valCollections"
        ]
        return self._val_collections

    @val_collections.setter
    def val_collections(self, value: Optional[Union[List[int], List[str]]]):
        if not isinstance(value, list):
            raise TypeError("val_collections must be a list of collection IDs or names.")
        self._val_collections = self._validate_collections(value)
        StateJson()[self.widget_id]["form"]["trainValSplit"][
            "valCollections"
        ] = self._val_collections
        StateJson().send_changes()

    @property
    def random_train_percentage(self) -> int:
        self._random_train_percentage = StateJson()[self.widget_id]["form"]["trainValSplit"][
            "randomTrainPercentage"
        ]
        return self._random_train_percentage

    @random_train_percentage.setter
    def random_train_percentage(self, value: int):
        if not isinstance(value, int) or not (0 <= value <= 100):
            raise ValueError("random_train_percentage must be an integer between 0 and 100.")
        self._random_train_percentage = value
        StateJson()[self.widget_id]["form"]["trainValSplit"]["randomTrainPercentage"] = value
        StateJson().send_changes()

    @property
    def train_val_split_mode(self) -> str:
        self._train_val_split_mode = StateJson()[self.widget_id]["form"]["trainValSplit"]["mode"]
        return self._train_val_split_mode

    @train_val_split_mode.setter
    def train_val_split_mode(self, value: str):
        self._train_val_split_mode = self._validate_split_mode(value)
        StateJson()[self.widget_id]["form"]["trainValSplit"]["mode"] = value
        StateJson().send_changes()

    def visible_changed(self, func):
        route_path = self.get_route_path(NewExperiment.Routes.VISIBLE_CHANGED)
        server = self._sly_app.get_server()
        self._visible_handled = True

        @server.post(route_path)
        def _click():
            func(self.visible)

        return _click

    @property
    def task_id(self) -> Optional[int]:
        return StateJson()[self.widget_id].get("taskId", None)

    @property
    def app_id(self) -> Optional[int]:
        return StateJson()[self.widget_id].get("appId", None)

    @property
    def model_id(self) -> Optional[int]:
        return StateJson()[self.widget_id].get("modelId", None)

    def app_started(self, func):
        route_path = self.get_route_path(NewExperiment.Routes.APP_STARTED)
        server = self._sly_app.get_server()
        self._app_started_handled = True

        @server.post(route_path)
        def _app_started():
            func(self.app_id, self.model_id, self.task_id)

        return _app_started

    def get_train_settings(self):
        train_settings = {
            "cvTask": self.cv_task,
            "projectId": self.project_id,
            "classes": self.classes,
            "trainValSplit": {
                "mode": self.train_val_split_mode,
                "randomTrainPercentage": self.random_train_percentage,
                "trainDatasets": self.training_datasets,
                "valDatasets": self.val_datasets,
                "trainCollections": self.train_collections,
                "valCollections": self.val_collections,
            },
            "modelId": self.model_id,
            "agentId": self.agent_id,
            "export": self.export,
            "runEvaluation": self.run_evaluation,
            "runSpeedTest": self.run_speed_test,
            "experimentName": self.experiment_name,
        }
        return train_settings
