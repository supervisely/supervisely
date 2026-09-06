import json
import os
from copy import deepcopy

from supervisely.imaging import image as sly_image
from supervisely.nn.legacy.inference.rest_constants import (
    GET_OUTPUT_META,
    IMAGE,
    INFERENCE,
    MODEL,
    OUTPUT_META,
    ANNOTATION,
    META,
    MODE,
    GPU_DEVICE,
)

from supervisely.worker_api.interfaces import SingleImageInferenceInterface
from supervisely.project.project_meta import ProjectMeta
from supervisely.nn.legacy.hosted.deploy import ModelDeploy
from supervisely.function_wrapper import function_wrapper

from flask import Flask
from flask_restful import Resource, Api, reqparse


class RestInferenceServer:
    """Simple Flask REST server that exposes legacy inference endpoints for a single-image model."""

    def __init__(self, model: SingleImageInferenceInterface, name, port=None):
        """
        :param model: Single image inference interface.
        :type model: :class:`~supervisely.worker_api.interfaces.SingleImageInferenceInterface`
        :param name: Flask app name.
        :type name: str
        :param port: Optional port.
        :type port: int
        """
        self._app = Flask(name)
        if port == '':
            port = None
        self._port = port

        api = Api(self._app)
        api.add_resource(RestInferenceServer.GetOutputMeta,
                         '/' + MODEL + '/' + GET_OUTPUT_META,
                         resource_class_kwargs={'model': model})
        api.add_resource(RestInferenceServer.Inference,
                         '/' + MODEL + '/' + INFERENCE,
                         resource_class_kwargs={'model': model})

    def run(self):
        self._app.run(debug=False, port=self._port, host='0.0.0.0')

    class GetOutputMeta(Resource):
        """Endpoint handler that returns model output metadata (`out_meta`) for a given input meta/mode."""

        def __init__(self, model):
            """See RestInferenceServer for model param."""
            self._model = model
            self._parser = reqparse.RequestParser()
            self._parser.add_argument(META)
            self._parser.add_argument(MODE)

        def post(self):
            args = self._parser.parse_args()

            meta = args[META]
            mode = args[MODE]

            data = {
                "request_type": GET_OUTPUT_META,
                "meta": json.loads(meta) if meta is not None else ProjectMeta().to_json(),
                "mode": json.loads(mode) if mode is not None else {}
            }

            response_json = self._model._final_processing(data)
            if 'out_meta' in response_json:
                return response_json['out_meta']
            if 'output_meta' in response_json:
                return response_json['output_meta']
            return response_json

    class Inference(Resource):
        """Endpoint handler that runs inference on an uploaded image and returns prediction JSON."""

        def __init__(self, model):
            """See RestInferenceServer for model param."""
            from werkzeug.datastructures import FileStorage
            self._model = model
            self._parser = reqparse.RequestParser()
            self._parser.add_argument(IMAGE, type=FileStorage, location='files', help="input image", required=True)
            self._parser.add_argument(ANNOTATION, location='files', type=FileStorage)
            self._parser.add_argument(META, location='files', type=FileStorage)
            self._parser.add_argument(MODE, location='files', type=FileStorage)

        def post(self):
            args = self._parser.parse_args()
            img_bytes = args[IMAGE].stream.read()
            img = sly_image.read_bytes(img_bytes)

            meta = args[META]
            ann = args[ANNOTATION]
            mode = args[MODE]

            data = {
                "request_type": INFERENCE,
                "meta": json.loads(meta.stream.read().decode("utf-8")) if meta is not None else ProjectMeta().to_json(),
                "annotation": json.loads(ann.stream.read().decode("utf-8")) if ann is not None else None,
                "mode": json.loads(mode.stream.read().decode("utf-8")) if mode is not None else {},
                'image_arr': img
            }
            return self._model._final_processing(data)


class ModelRest(ModelDeploy):
    """Legacy deployment wrapper that configures a model for REST inference (not runnable as a standalone app)."""

    def load_config(self):
        gpu_device = os.getenv(GPU_DEVICE, 0)
        self.config = deepcopy(ModelDeploy.config)
        self.config["model"]["gpu_device"] = gpu_device

    def run(self):
        raise RuntimeError("Method is not accessible")

    def _create_applier(self):
        model_applier = function_wrapper(self.model_applier_cls, task_model_config={})
        return model_applier
