from typing import Dict, List, Optional

from fastapi import BackgroundTasks, Request

import supervisely.io.fs as sly_fs
from supervisely._utils import logger
from supervisely.api.api import Api
from supervisely.app.fastapi.subapp import Application
from supervisely.nn.inference.predict_app.gui.gui import PredictAppGui
from supervisely.nn.inference.predict_app.gui.utils import disable_enable
from supervisely.nn.model.prediction import Prediction


class PredictApp:
    def __init__(self, api: Api):
        _static_dir = "static"
        sly_fs.mkdir(_static_dir, True)
        self.api = api
        self.gui = PredictAppGui(api, static_dir=_static_dir)
        self.app = Application(self.gui.layout, static_dir=_static_dir)
        self._add_endpoints()

        @self.gui.output_selector.start_button.click
        def start_prediction():
            if self.gui.output_selector.validate_step():
                widgets_to_disable = self.gui.output_selector.widgets_to_disable + [self.gui.settings_selector.preview.run_button]
                disable_enable(widgets_to_disable, True)
                self.gui.run()
                self.shutdown_serving_app()
                self.shutdown_predict_app()

    def shutdown_serving_app(self):
        if self.gui.output_selector.should_stop_serving_on_finish():
            logger.info("Stopping serving app...")
            self.gui.model_selector.model.stop()

    def shutdown_predict_app(self):
        if self.gui.output_selector.should_stop_self_on_finish():
            self.gui.output_selector.start_button.disable()
            logger.info("Stopping Predict App...")
            self.app.stop()
        else:
            disable_enable(self.gui.output_selector.widgets_to_disable, False)
            self.gui.output_selector.start_button.enable()

    def run(self, run_parameters: Optional[Dict] = None) -> List[Prediction]:
        return self.gui.run(run_parameters)

    def stop(self):
        self.gui.stop()

    def shutdown_model(self):
        self.gui.shutdown_model()

    def load_from_json(self, data):
        self.gui.load_from_json(data)
        if data.get("run", False):
            try:
                self.run()
            except Exception as e:
                raise
            finally:
                if data.get("stop_after_run", False):
                    self.shutdown_model()
                    self.app.stop()

    def get_inference_settings(self):
        return self.gui.settings_selector.get_inference_settings()

    def get_run_parameters(self):
        return self.gui.get_run_parameters()

    def _add_endpoints(self):
        server = self.app.get_server()

        @server.post("/load")
        def load(request: Request, background_tasks: BackgroundTasks):
            """
            Load the model state from a JSON object.
            This endpoint initializes the model with the provided state.
            All the fields are optional

            Example state:
                state = {
                    "model": {
                        "mode": "connect",
                        "session_id": "12345"
                        # "mode": "pretrained",
                        # "framework: "YOLO",
                        # "model_name": "YOLO11m-seg",
                        # "mode": "custom",
                        # "train_task_id": 123
                    },
                    "input": {
                        "project_id": 123,
                        # "dataset_ids": [...],
                        # "video_id": 123
                    },
                    "settings": {
                        "inference_settings": {
                            "confidence_threshold": 0.5
                        },
                    }
                    "output": {
                        "mode": "create",
                        "project_name": "Predictions",
                        # "mode": "append",
                        # "mode": "replace",
                        # "mode": "iou_merge",
                        # "iou_merge_threshold": 0.5
                    }
                }
            """
            state = request.state.state
            stop_after_run = state.get("stop_after_run", False)
            if stop_after_run:
                state["stop_after_run"] = False
            self.load_from_json(state)
            if stop_after_run:
                self.shutdown_model()
                background_tasks.add_task(self.app.stop)

        @server.post("/deploy")
        def deploy(request: Request):
            """
            Deploy the model for inference.
            This endpoint prepares the model for running predictions.
            """
            self.gui.model_selector.model._deploy()

        @server.get("/inference_settings")
        def get_inference_settings():
            """
            Get the inference settings for the model.
            This endpoint returns the current inference settings.
            """
            return self.get_inference_settings()

        @server.get("/run_parameters")
        def get_run_parameters():
            """
            Get the run parameters for the model.
            This endpoint returns the parameters needed to run the model.
            """
            return self.get_run_parameters()

        @server.post("/predict")
        def predict(request: Request):
            """
            Run the model prediction.
            This endpoint processes the request data and runs the model prediction.

            Example data:
                data = {
                    "inference_settings": {
                        "conf": 0.6,
                    },
                    "input": {
                        # "project_id": ...,
                        # "dataset_ids": [...],
                        "image_ids": [1148679, 1148675],
                    },
                    "output": {"mode": "iou_merge", "iou_merge_threshold": 0.5},
                }
            """
            state = request.state.state
            predictions = self.run(state)
            return [prediction.to_json() for prediction in predictions]

        @server.post("/run")
        def run(request: Request):
            """
            Run the model prediction.
            """
            predicitons = self.run()
            return [prediction.to_json() for prediction in predicitons]
