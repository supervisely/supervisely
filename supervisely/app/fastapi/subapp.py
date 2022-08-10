import os
import signal
import functools
import asyncio
import psutil
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# from async_asgi_testclient import TestClient

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
from supervisely._utils import is_production
from supervisely.app.fastapi.offline import dump_files_to_supervisely


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
        # TODO:
        # async def goodbye(delay=0.01):
        #     await asyncio.sleep(delay)
        # # run_sync
        # task1 = asyncio.create_task(goodbye(1))
        # # await task1

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
    from supervisely.app.fastapi import available_after_shutdown

    if app is None:
        app = FastAPI()
    Jinja2Templates(directory=[Path(__file__).parent.absolute(), templates_dir])
    enable_hot_reload_on_debug(app)
    app.mount("/sly", create())
    handle_server_errors(app)

    # @app.middleware("http")
    # async def get_state_from_request(request: Request, call_next):
    #     from supervisely.app.content import StateJson

    #     await StateJson.from_request(request)
    #     response = await call_next(request)
    #     return response

    # @available_after_shutdown(app)
    @app.get("/")
    def read_index(request: Request = None):
        return Jinja2Templates().TemplateResponse("index.html", {"request": request})

    # @app.on_event("shutdown")
    # def shutdown():
    #     # try:
    #     #     client = TestClient(app)
    #     #     responce = client.get("/")
    #     # except Exception as e:
    #     #     print(repr(e))
    #     x = 10
    #     x += 1
    #     # read_index(Request())  # save last version of static files

    # @app.on_event("shutdown")
    # async def shutdown():
    #     read_index()
    #     # async with TestClient(app) as client:
    #     #     response = await client.get("/")
    #     #     logger.debug(
    #     #         f"shutdown event: response.status_code == {response.status_code}"
    #     #     )
    #     #     print(111)

    return app


class Application(metaclass=Singleton):
    def __init__(self, name="", templates_dir: str = "templates"):
        self._fastapi: FastAPI = _init(app=None, templates_dir=templates_dir)

    def get_server(self):
        return self._fastapi

    async def __call__(self, scope, receive, send) -> None:
        await self._fastapi.__call__(scope, receive, send)

    def shutdown(self):
        from supervisely.app.fastapi import shutdown

        shutdown()
