from supervisely import Application
from supervisely.app.widgets import Widget
from supervisely.nn.training.gui.gui import TrainGUI


class TrainApp:
    def __init__(
        self,
        models: str | list,
        hyperparameters: str,
        app_options: str | dict = None,
    ):
        # self.project_id = None
        # self.train_dataset_id = None
        # self.val_dataset_id = None
        # self.task_type: str = None
        # self.selected_model: dict = None
        self._layout: TrainGUI = TrainGUI(models, hyperparameters, app_options)
        self._app = Application(layout=self._layout.layout)
        self._server = self._app.get_server()
        
    @property
    def app(self) -> Application:
        return self._app

    @property
    def project_id(self) -> int:
        return self._layout.project_id

    @property
    def train_dataset_id(self) -> int:
        return self._layout.input_data.get_train_dataset_id()

    @property
    def val_dataset_id(self) -> int:
        return self._layout.input_data.get_val_dataset_id()
