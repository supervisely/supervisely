from __future__ import annotations
from fastapi import FastAPI
from supervisely.app.singleton import Singleton
import supervisely.app.fastapi as fastapi


class App(metaclass=Singleton):
    def __init__(self, templates_dir: str = "templates") -> App:
        self._fastapi: FastAPI = fastapi.init(app=None, templates_dir=templates_dir)

    def get_server(self):
        return self._fastapi
