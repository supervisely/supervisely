import asyncio
import os
import signal
import psutil
import sys
from pathlib import Path
import concurrent.futures

from fastapi import (
    FastAPI,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
)
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


# async def goodbue():
#     await asyncio.sleep(0.2)


def shutdown():
    # shutdown fastapi
    try:
        logger.info("Shutting down...")
        # process_id = psutil.Process(os.getpid()).ppid()
        current_process = psutil.Process(os.getpid())
        # current_process = psutil.Process(process_id)
        current_process.send_signal(signal.SIGINT)  # emit ctrl + c
    except KeyboardInterrupt:
        logger.info("Application has been shut down successfully")


def shutdown_parent():
    # shutdown uvicorn after fastapi
    try:
        current_process = psutil.Process(os.getpid())
        current_process.send_signal(signal.SIGINT)  # emit ctrl + c
    except KeyboardInterrupt:
        pass


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

    @app.get("/")
    @available_after_shutdown(app)
    def read_index(request: Request):
        return Jinja2Templates().TemplateResponse("index.html", {"request": request})

    @app.on_event("shutdown")
    def shutdown():
        client = TestClient(app)
        responce = client.get("/")
        # shutdown_parent()

    return app


class Application(metaclass=Singleton):
    def __init__(self, name="", templates_dir: str = "templates"):
        self._fastapi: FastAPI = _init(app=None, templates_dir=templates_dir)
        self._run_executors()

    def get_server(self):
        return self._fastapi

    async def __call__(self, scope, receive, send) -> None:
        await self._fastapi.__call__(scope, receive, send)

    def shutdown(self):
        client = TestClient(self.get_server())
        try:
            responce = client.post("/sly/shutdown")
        except KeyboardInterrupt:
            pass

    async def _shutdown(self, signal=None, error=None):
        """Cleanup tasks tied to the service's shutdown."""
        if signal:
            self.logger.info(f"Received exit signal {signal.name}...")
        self.logger.info("Nacking outstanding messages")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        [task.cancel() for task in tasks]

        self.logger.info(f"Cancelling {len(tasks)} outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info("Shutting down ThreadPoolExecutor")
        self.executor.shutdown(wait=False)

        self.logger.info(
            f"Releasing {len(self.executor._threads)} threads from executor"
        )
        for thread in self.executor._threads:
            try:
                thread._tstate_lock.release()
            except Exception:
                pass

        self.logger.info(f"Flushing metrics")
        self.loop.stop()

        if error is not None:
            self._error = error

    def _run_executors(self):
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.loop = asyncio.get_event_loop()
        # May want to catch other signals too
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)
        for s in signals:
            self.loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(self._shutdown(signal=s))
            )
        # comment out the line below to see how unhandled exceptions behave
        self.loop.set_exception_handler(self.handle_exception)

    def handle_exception(self, loop, context):
        # context["message"] will always be there; but context["exception"] may not
        msg = context.get("exception", context["message"])
        if isinstance(msg, Exception):
            # self.logger.error(traceback.format_exc(), exc_info=True, extra={'exc_str': str(msg), 'future_info': context["future"]})
            self.logger.error(
                msg, exc_info=True, extra={"future_info": context["future"]}
            )
        else:
            self.logger.error("Caught exception: {}".format(msg))

        self.logger.info("Shutting down...")
        asyncio.create_task(self._shutdown())
