from fastapi import FastAPI
import jsonpatch
# from fastapi import Request
from supervisely.fastapi_helpers.websocket import WebsocketManager


class PatchableJson(dict):
    _last = {}
    _app = None
    _ws = None
    
    def __init__(self, app: FastAPI, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._app = app

    def _get_patch(self):
        patch = jsonpatch.JsonPatch.from_diff(self._last, self)
        return patch

    def _apply_patch(self, patch):
        patch.apply(self._last, in_place=True)



