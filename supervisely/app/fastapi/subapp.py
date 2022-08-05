import os
import signal
import psutil
import sys
from pathlib import Path

from fastapi import (
    FastAPI,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
)
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from supervisely.app.singleton import Singleton

from supervisely.app.fastapi.templating import Jinja2Templates
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.io.fs import mkdir, dir_exists
from supervisely.sly_logger import logger
from supervisely.api.api import SERVER_ADDRESS, API_TOKEN, TASK_ID, Api


def create() -> FastAPI:
    from supervisely.app import DataJson, StateJson

    app = FastAPI()
    WebsocketManager().set_app(app)

    @app.post("/data")
    async def send_data(request: Request):
        data = DataJson()
        response = JSONResponse(content=dict(data))
        return response

    @app.post("/state")
    async def send_state(request: Request):
        state = StateJson()
        response = JSONResponse(content=dict(state))

        gettrace = getattr(sys, "gettrace", None)
        if gettrace is not None and gettrace():
            response.headers["x-debug-mode"] = "1"
        return response

    @app.post("/session-info")
    async def send_session_info(request: Request):
        server_address = os.environ.get(SERVER_ADDRESS)
        if server_address is not None:
            server_address = Api.normalize_server_address(server_address)

        response = JSONResponse(
            content={
                TASK_ID: os.environ.get(TASK_ID),
                SERVER_ADDRESS: server_address,
                API_TOKEN: os.environ.get(API_TOKEN),
            }
        )
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

    import supervisely

    app.mount("/css", StaticFiles(directory=supervisely.__path__[0]), name="sly_static")
    return app


def shutdown():
    try:
        logger.info("Shutting down...")
        current_process = psutil.Process(os.getpid())
        current_process.send_signal(signal.SIGINT)  # emit ctrl + c
    except KeyboardInterrupt:
        logger.info("Application has been shut down successfully")


def enable_hot_reload_on_debug(app: FastAPI):
    templates = Jinja2Templates()
    gettrace = getattr(sys, "gettrace", None)
    if gettrace is None:
        print("Can not detect debug mode, no sys.gettrace")
    elif gettrace():
        print("In debug mode ...")
        import arel

        hot_reload = arel.HotReload(paths=[arel.Path(".")])
        app.add_websocket_route("/hot-reload", route=hot_reload, name="hot-reload")
        app.add_event_handler("startup", hot_reload.startup)
        app.add_event_handler("shutdown", hot_reload.shutdown)
        templates.env.globals["DEBUG"] = "1"
        templates.env.globals["hot_reload"] = hot_reload
    else:
        print("In runtime mode ...")


def handle_server_errors(app: FastAPI):
    @app.exception_handler(500)
    async def server_exception_handler(request, exc):
        return await http_exception_handler(
            request,
            HTTPException(
                status_code=500,
                detail={
                    # "title": "error title",
                    "message": repr(exc)
                },
            ),
        )


def _init(app: FastAPI = None, templates_dir: str = "templates") -> FastAPI:
    if app is None:
        app = FastAPI()
    Jinja2Templates(directory=[Path(__file__).parent.absolute(), templates_dir])
    enable_hot_reload_on_debug(app)
    app.mount("/sly", create())
    handle_server_errors(app)

    @app.get("/")
    async def read_index(request: Request):
        return Jinja2Templates().TemplateResponse("index.html", {"request": request})

    return app


class Application(metaclass=Singleton):
    def __init__(self, name="", templates_dir: str = "templates"):
        self._fastapi: FastAPI = _init(app=None, templates_dir=templates_dir)

    def get_server(self):
        return self._fastapi

    async def __call__(self, scope, receive, send) -> None:
        await self._fastapi.__call__(scope, receive, send)
