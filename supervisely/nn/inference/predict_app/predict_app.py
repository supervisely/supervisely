from supervisely.api.api import Api
from supervisely.app.fastapi.subapp import Application
from supervisely.nn.inference.predict_app.gui import PredictAppGui


class PredictApp:
    def __init__(self, api: Api):
        _static_dir = "static"
        self.api = api
        self.gui = PredictAppGui(api, static_dir=_static_dir)
        self.app = Application(self.gui.layout, static_dir=_static_dir)

    def load_from_json(self, data):
        self.gui.load_from_json(data)

    def get_inference_settings(self):
        return self.gui.get_inference_settings()

    def get_run_parameters(self):
        return self.gui.get_run_parameters()
