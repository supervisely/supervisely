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
import jinja2
from fastapi.testclient import TestClient
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from supervisely.app.fastapi.utils import run_sync

from supervisely.app.singleton import Singleton

from supervisely.app.fastapi.templating import Jinja2Templates
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.io.fs import mkdir, dir_exists
from supervisely.sly_logger import logger
from supervisely.api.api import SERVER_ADDRESS, API_TOKEN, TASK_ID, Api
from supervisely._utils import is_production, is_development
from async_asgi_testclient import TestClient


def create(process_id=None, headless=False) -> FastAPI:
    from supervisely.app import DataJson, StateJson

    app = FastAPI()
    WebsocketManager().set_app(app)

    @app.post("/shutdown")
    async def shutdown_endpoint(request: Request):
        shutdown(process_id)

    if headless is False:

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

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await WebsocketManager().connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
            except WebSocketDisconnect:
                WebsocketManager().disconnect(websocket)

        import supervisely

        app.mount(
            "/css", StaticFiles(directory=supervisely.__path__[0]), name="sly_static"
        )

    return app


def shutdown(process_id=None):
    try:
        logger.info(f"Shutting down [pid argument = {process_id}]...")
        if process_id is None:
            # process_id = psutil.Process(os.getpid()).ppid()
            process_id = os.getpid()
        current_process = psutil.Process(process_id)
        current_process.send_signal(signal.SIGINT)  # emit ctrl + c
    except KeyboardInterrupt:
        logger.info("Application has been shut down successfully")


def enable_hot_reload_on_debug(app: FastAPI):
    templates = Jinja2Templates()
    gettrace = getattr(sys, "gettrace", None)
    if gettrace is None:
        print("Can not detect debug mode, no sys.gettrace")
    elif gettrace():
        logger.debug("In debug mode ...")
        import arel

        hot_reload = arel.HotReload(paths=[arel.Path(".")])
        app.add_websocket_route("/hot-reload", route=hot_reload, name="hot-reload")
        app.add_event_handler("startup", hot_reload.startup)
        app.add_event_handler("shutdown", hot_reload.shutdown)
        templates.env.globals["DEBUG"] = "1"
        templates.env.globals["hot_reload"] = hot_reload
    else:
        logger.debug("In runtime mode ...")


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


def _init(
    app: FastAPI = None,
    templates_dir: str = "templates",
    headless=False,
    process_id=None,
) -> FastAPI:
    from supervisely.app.fastapi import available_after_shutdown
    from supervisely.app.content import StateJson

    if app is None:
        app = FastAPI()

    handle_server_errors(app)

    if headless is False:
        if "app_body_padding" not in StateJson():
            StateJson()["app_body_padding"] = "20px"
        Jinja2Templates(
            directory=[str(Path(__file__).parent.absolute()), templates_dir]
        )
        enable_hot_reload_on_debug(app)

    app.mount("/sly", create(process_id, headless))

    if headless is False:

        @app.middleware("http")
        async def get_state_from_request(request: Request, call_next):
            await StateJson.from_request(request)
            response = await call_next(request)
            return response

        @app.get("/")
        @available_after_shutdown(app)
        def read_index(request: Request):

            try:
                template = Jinja2Templates().TemplateResponse(
                    "index.html", {"request": request}
                )
            except jinja2.exceptions.TemplateNotFound as e:
                xx = 10
            return template

        @app.on_event("shutdown")
        def shutdown():
            client = TestClient(app)
            resp = run_sync(client.get("/"))
            assert resp.status_code == 200
            logger.info("Application has been shut down successfully")

    return app


class Application(metaclass=Singleton):
    def __init__(self, templates_dir: str = "templates", headless=False):
        if is_production():
            logger.info(
                "Application is running on Supervisely Platform in production mode"
            )
        else:
            logger.info("Application is running on localhost in development mode")
        self._process_id = os.getpid()
        logger.info(f"Application PID is {self._process_id}")
        self._fastapi: FastAPI = _init(
            app=None,
            templates_dir=templates_dir,
            headless=headless,
            process_id=self._process_id,
        )

    def get_server(self):
        return self._fastapi

    async def __call__(self, scope, receive, send) -> None:
        await self._fastapi.__call__(scope, receive, send)

    def shutdown(self):
        shutdown(self._process_id)


def get_name_from_env(default="Supervisely App"):
    return os.environ.get("APP_NAME", default)
