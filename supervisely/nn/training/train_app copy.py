from supervisely.app.widgets import Widget

# class in SDK:
class TrainApp:

    # required:
    # training_parameters.yaml
    # models.json

    # optional:
    # app_options.yaml

    def __init__(
            self,
            models: str | list,
            hyperparameters: str,
            app_options: str | dict = None,
        ):
        """
        Args:
            models: str | list
                Path to models.json or list of models
            hyperparameters_yaml: str
                Path to hyperparameters.yaml
            app_options: str | dict
                Path to app_options.yaml or dict
        """
        # self.project_id = None
        # self.train_dataset_id = None
        # self.val_dataset_id = None
        # self.task_type: str = None
        # self.selected_model: dict = None
        self._layout: TraininingLayout = None
        
    @property
    def project_id(self) -> int:
        return self._layout.project_selector.project_id

    @property
    def train_dataset_id(self) -> int:
        return self._layout.project_selector.train_dataset_id
    
    @property
    def val_dataset_id(self) -> int:
        return self._layout.project_selector.val_dataset_id
    
    def get_hyperparameters(self) -> dict:
        return self._layout.get_hyperparameters()

    def load_app_config(self, config: dict):
        # config = {
        #     "project_id": 123,
        #     "train_dataset_id": 456,
        #     "val_dataset_id": 789,
        #     "model": "yolov8s-det",
        #     "classes": ["car", "person", "dog"],
        # }
        self._layout.load_app_config(config)
        # ...


    def _build_layout(self):
        # 1. Project selection
        #    Train/val split
        # 2. Task type (optional, auto-detect)
        #    Model selection
        # 3. Select classes
        # 4. Training parameters (yaml), scheduler preview
        # 5. Other options
        # 6. Start training button / Stop
        # 7. Progress + charts (tensorboard frame)
        # 8. Upload checkpoints
        # 9. Evaluation report
        self._layout = TraininingLayout(
            models=self.models,
            hyperparameters=self.hyperparameters,
            app_options=self.app_options
        )
        @self._layout.on_train_start
        def on_train_start():
            self.on_train_start()
        pass

    def on_train_start(self, func):
        # крутой декаратор
        self._train_func = func
        # on stop callback
        try:
            func()
        except StopTrainingException as e:
            # train stopping
            # uploading checkpoints
            pass
        self.workflow()

    def auto_train(self, config: dict):
        self.load_app_config(config)
        self._train_func()  # TODO

    def serve(self):
        from supervisely import Application
        app = Application(layout=self._layout)

        server = app.get_server()
        @server.post("/auto_train")
        def auto_train(state: dict):
            config = get_config(state)
            self.auto_train(config)
            return {"status": "ok"}

    def workflow(self):
        pass


# trainining_layout.py
class TraininingLayout:
    def __init__(
            self,
            models: list,
            hyperparameters: str,
            app_options: dict = None,
            ):
        self.layout: Widget = None
        self.project_selector: ProjectSelector = None
        self._small_widget = ...
    
        @self.select_btn_project
        def on_select_btn_project(self, project_id: int):
            self.set_project(project_id)

    def set_project(self, project_id: int):
        self.project_selector.set_project(project_id)


# main.py
train_app = TrainApp(
    hyperparameters_yaml: str,
)

@train_app.on_train_start
def on_train_start():
    training_parameters: dict = train_app.get_hyperparameters()
    selected_checkpoint: dict = train_app.get_selected_checkpoint()

    # model specific code
    convert_sly2yolov8(train_app.project_id)
    config = load_config(selected_checkpoint["config_path"])
    model = model(config)
    trainer = Trainer(model, training_parameters)
    trainer.train()
    # ...

    # in train_loop
    import sly_aug
    import sly_train_logger
    image_np, ann = sly_aug.process(image_np_or_pil, ann_coco_or_xxx)  # TODO
    loss = model.backward()
    sly_train_logger.log({
        "Train/loss": loss,
        "Train/loss2": loss2,
        "Val/accuracy": accuracy,
        })
    # ...

    train_app.upload_checkpoints(dir_path="checkpoints")
    train_app.upload(checkpoint_files, checkpoint_infos)
    # train_app adds them to pandas / parquet / pyarrow / DB
    train_app.evaluate_model_benchmark()


@train_app.on_upload_checkpoints
def prepare_uploading_checkpoints():
    # подумать на второй итерации
    # юзер выбирает какие чекпоинты нужно загрузить
    # может быть проапрегрейдим sly.artifacts
    os.makedirs("checkpoints")
    os.move("checkpoint.pt", "checkpoints")
    # ...