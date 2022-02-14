import os
import signal
import psutil
import sys

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from supervisely.app.fastapi.templating import Jinja2Templates
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.io.fs import mkdir, dir_exists
from supervisely.sly_logger import logger

# "--reload-include", "*.py,*.html"


def create() -> FastAPI:
    from supervisely.app import DataJson, LastStateJson

    app = FastAPI()
    WebsocketManager().set_app(app)

    @app.post("/data")
    async def send_data(request: Request):
        data = DataJson()
        response = JSONResponse(content=dict(data))
        return response

    @app.post("/state")
    async def send_state(request: Request):
        state = LastStateJson()
        response = JSONResponse(content=dict(state))

        gettrace = getattr(sys, "gettrace", None)
        if gettrace is not None and gettrace():
            response.headers["x-debug-mode"] = "1"
        return response

    @app.post("/shutdown")
    async def shutdown_endpoint(request: Request):
        shutdown()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await WebsocketManager().connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
        except WebSocketDisconnect:
            WebsocketManager().disconnect(websocket)

    return app


def shutdown():
    current_process = psutil.Process(os.getpid())
    current_process.send_signal(signal.SIGINT)  # emit ctrl + c


def enable_hot_reload_on_debug(app: FastAPI, templates: Jinja2Templates):
    gettrace = getattr(sys, "gettrace", None)
    if gettrace is None:
        print("Can not detect debug mode, no sys.gettrace")
    elif gettrace() and templates is not None:
        print("In debug mode ...")
        import arel

        hot_reload = arel.HotReload(paths=[arel.Path(".")])
        app.add_websocket_route("/hot-reload", route=hot_reload, name="sly-hot-reload")
        app.add_event_handler("startup", hot_reload.startup)
        app.add_event_handler("shutdown", hot_reload.shutdown)
        templates.env.globals["DEBUG"] = "1"
        templates.env.globals["hot_reload"] = hot_reload
    else:
        print("In runtime mode ...")
