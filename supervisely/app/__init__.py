from __future__ import annotations
import sys

_import_failed = False
try:
    from fastapi import FastAPI
    from supervisely.app.content import StateJson, DataJson
    from supervisely.app.content import get_data_dir
    import supervisely.app.fastapi as fastapi
    import supervisely.app.widgets as widgets
except (ImportError, ModuleNotFoundError) as e:
    print(repr(e))
    _import_failed = True
    pass


def __getattr__(name):
    if _import_failed is True:
        raise ModuleNotFoundError(
            'No module named supervisely.app, please install dependencies with "pip install supervisely[apps]"'
        )
    return getattr(sys.modules[__name__], name)


from supervisely.app.singleton import Singleton


class Application(metaclass=Singleton):
    def __init__(self, templates_dir: str = "templates"):
        self._fastapi: FastAPI = fastapi.init(app=None, templates_dir=templates_dir)

    def get_server(self):
        return self._fastapi
