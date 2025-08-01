from fastapi import Request

from supervisely.api.api import Api
from supervisely.app.fastapi.subapp import Application
from supervisely.nn.inference.predict_app.gui import PredictAppGui


class PredictApp:
    def __init__(self, api: Api):
        _static_dir = "static"
        self.api = api
        self.gui = PredictAppGui(api, static_dir=_static_dir)
        self.app = Application(self.gui.layout, static_dir=_static_dir)
        self._add_endpoints()

    def load_from_json(self, data):
        self.gui.load_from_json(data)

    def get_inference_settings(self):
        return self.gui.get_inference_settings()

    def get_run_parameters(self):
        return self.gui.get_run_parameters()

    def _add_endpoints(self):
        server = self.app.get_server()

        @server.post("/load")
        def load(request: Request):
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
                    "items": {
                        "project_id": 123,
                        # "dataset_ids": [...],
                        # "video_id": 123
                    },
                    "inference_settings": {
                        "confidence_threshold": 0.5
                    },
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
            self.load_from_json(state)

        @server.post("/deploy")
        def deploy(request: Request):
            """
            Deploy the model for inference.
            This endpoint prepares the model for running predictions.
            """
            self.gui.model._deploy()

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
                    "item": {
                        # "project_id": ...,
                        # "dataset_ids": [...],
                        "image_ids": [1148679, 1148675],
                    },
                    "output": {"mode": "iou_merge", "iou_merge_threshold": 0.5},
                }
            """
            state = request.state.state
            run_parameters = {
                "item": state["item"],
            }
            if "inference_settings" in state:
                run_parameters["inference_settings"] = state["inference_settings"]
            if "output" in state:
                run_parameters["output"] = state["output"]
            else:
                run_parameters["output"] = {"mode": None}

            predictions = self.gui.run(run_parameters)
            return [prediction.to_json() for prediction in predictions]

        @server.post("/run")
        def run(request: Request):
            """
            Run the model prediction.
            """
            predicitons = self.gui._run()
            return [prediction.to_json() for prediction in predicitons]
