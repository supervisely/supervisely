from supervisely._utils import logger
from supervisely.api.api import Api
from supervisely.app.fastapi.subapp import Application
from supervisely.nn.inference.predict_app.gui import PredictAppGui


class PredictApp:
    def __init__(self, api: Api):
        self.api = api
        self.gui = PredictAppGui(api)
        self.app = Application(self.gui.layout)

        @self.gui.run_button.click
        def run_button_click():
            self.run()

    def run(self, config=None):
        if config is None:
            run_parameters = self.gui.get_run_parameters()
        else:
            run_parameters = config

        if self.gui.model.model_api is None:
            self.gui.model.deploy()

        model_api = self.gui.model.model_api
        if model_api is None:
            logger.error("Model Deployed with an error")
            return

        inference_settings = run_parameters["inference_settings"]
        if not inference_settings:
            inference_settings = {}

        item_prameters = run_parameters["item"]

        output_parameters = run_parameters["output"]
        upload_parameters = {}
        upload_mode = output_parameters["mode"]

        upload_parameters["upload_mode"] = upload_mode
        if upload_mode == "iou_merge":
            upload_parameters["existing_objects_iou_thresh"] = output_parameters[
                "iou_merge_threshold"
            ]

        model_api.predict(
            **item_prameters, **inference_settings, **upload_parameters, tqdm=self.gui.progress()
        )

    def load_from_json(self, data):
        self.gui.load_from_json(data)
