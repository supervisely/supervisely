from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.imaging import image as sly_image
from supervisely_lib.nn.inference.rest_constants import GET_OUTPUT_META, IMAGE, INFERENCE, MODEL, OUTPUT_META, ANNOTATION
from supervisely_lib.worker_api.interfaces import SingleImageInferenceInterface

from flask import Flask, request
from flask_restful import Resource, Api, reqparse

import werkzeug


class RestInferenceServer:
    def __init__(self, model: SingleImageInferenceInterface, name, port=None):
        self._app = Flask(name)
        self._port = port

        api = Api(self._app)
        api.add_resource(RestInferenceServer.GetOutputMeta,
                         '/' + MODEL + '/' + GET_OUTPUT_META,
                         resource_class_kwargs={'out_meta_json': model.get_out_meta().to_json()})
        api.add_resource(RestInferenceServer.Inference,
                         '/' + MODEL + '/' + INFERENCE,
                         resource_class_kwargs={'model': model})

    def run(self):
        self._app.run(debug=False, port=self._port)

    class GetOutputMeta(Resource):
        def __init__(self, out_meta_json):
            self._out_meta_json = out_meta_json

        def post(self):
            return {OUTPUT_META: self._out_meta_json}

    class Inference(Resource):
        def __init__(self, model: SingleImageInferenceInterface):
            self._model = model
            self._parser = reqparse.RequestParser()
            self._parser.add_argument(
                IMAGE, type=werkzeug.FileStorage, location='files', help="input image", required=True)

        def post(self):
            args = self._parser.parse_args()
            img_bytes = args[IMAGE].stream.read()
            img = sly_image.read_bytes(img_bytes)
            ann = self._model.inference(img, Annotation(img_size=img.shape[:2]))
            return {ANNOTATION: ann.to_json()}
