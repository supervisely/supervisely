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

    def run(self):
        run_parameters = self.gui.get_run_parameters()
        model_parameters = run_parameters["model"]
        params = model_parameters.get("params", {})
        model_api = self.gui.model.deploy()
        if model_api is None:
            logger.error("Model Deployed with an error")
            return

        item_prameters = run_parameters["item"]

        output_parameters = run_parameters["output"]
        upload_parameters = {}
        upload_mode = output_parameters["mode"]

        upload_parameters["upload_mode"] = upload_mode
        if upload_mode == "iou_merge":
            upload_parameters["existing_objects_iou_thresh"] = output_parameters[
                "iou_merge_threshold"
            ]

        model_api.predict(**item_prameters, **params, **upload_parameters, tqdm=self.gui.progress())
