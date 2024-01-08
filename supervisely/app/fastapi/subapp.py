import os
import signal
import sys
from contextlib import suppress
from functools import wraps
from pathlib import Path
from threading import Event as ThreadingEvent
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import arel
import jinja2
import numpy as np
import psutil
from async_asgi_testclient import TestClient
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import supervisely.io.env as sly_env
from supervisely._utils import (
    is_debug_with_sly_net,
    is_development,
    is_docker,
    is_production,
)
from supervisely.api.api import API_TOKEN, SERVER_ADDRESS, TASK_ID, Api
from supervisely.api.module_api import ApiField
from supervisely.app.exceptions import DialogWindowBase
from supervisely.app.fastapi.custom_static_files import CustomStaticFiles
from supervisely.app.fastapi.templating import Jinja2Templates
from supervisely.app.fastapi.utils import run_sync
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.app.singleton import Singleton
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.geometry.bitmap import Bitmap
from supervisely.io.fs import dir_exists, mkdir
from supervisely.sly_logger import logger

# from supervisely.app.fastapi.request import Request


if TYPE_CHECKING:
    from supervisely.app.widgets import Widget


class Event:
    class Brush:
        class DrawLeftMouseReleased:
            endpoint = "/tools_bitmap_brush_figure_changed"

            def __init__(
                self,
                team_id: int,
                workspace_id: int,
                project_id: int,
                dataset_id: int,
                image_id: int,
                label_id: int,
                object_class_id: int,
                object_class_title: str,
                tool_class_id: int,
                session_id: int,
                tool: str,
                user_id: int,
                job_id: int,
                is_fill: bool,
                is_erase: bool,
                geometry_type: str,
                mask: np.ndarray,
            ):
                self.dataset_id = dataset_id
                self.team_id = team_id
                self.workspace_id = workspace_id
                self.project_id = project_id
                self.image_id = image_id
                self.label_id = label_id
                self.object_class_id = object_class_id
                self.object_class_title = object_class_title
                self.tool_class_id = tool_class_id
                self.session_id = session_id
                self.tool = tool
                self.user_id = user_id
                self.job_id = job_id
                self.is_fill = is_fill
                self.is_erase = is_erase
                self.geometry_type = geometry_type
                self.mask = mask

            @classmethod
            def from_json(cls, data: Dict[str, Any]):
                tool_state = data.get(ApiField.TOOL_STATE)
                if tool_state is not None:
                    tool_option = tool_state.get(ApiField.OPTION)
                    if tool_option == "fill":
                        is_fill = True
                        is_erase = False
                    elif tool_option == "erase":
                        is_fill = False
                        is_erase = True
                else:
                    is_fill = False
                    is_erase = False
                geometry_type = None
                mask = None
                figure_state = data.get(ApiField.FIGURE_STATE)
                try:
                    geometry_type = figure_state.get(ApiField.GEOMETRY_TYPE)
                    geometry = figure_state.get(ApiField.GEOMETRY).get(ApiField.BITMAP)
                    mask = Bitmap.base64_2_data(geometry.get(ApiField.DATA))
                except AttributeError:
                    pass
                return cls(
                    team_id=data.get(ApiField.TEAM_ID),
                    workspace_id=data.get(ApiField.WORKSPACE_ID),
                    project_id=data.get(ApiField.PROJECT_ID),
                    dataset_id=data.get(ApiField.DATASET_ID),
                    image_id=data.get(ApiField.IMAGE_ID),
                    label_id=data.get(ApiField.FIGURE_ID),
                    object_class_id=data.get(ApiField.FIGURE_CLASS_ID),
                    object_class_title=data.get(ApiField.FIGURE_CLASS_TITLE),
                    tool_class_id=data.get(ApiField.TOOL_CLASS_ID),
                    session_id=data.get(ApiField.SESSION_ID),
                    tool=data.get(ApiField.LABELING_TOOL),
                    user_id=data.get(ApiField.USER_ID),
                    job_id=data.get(ApiField.JOB_ID),
                    is_fill=is_fill,
                    is_erase=is_erase,
                    geometry_type=geometry_type,
                    mask=mask,
                )

    class ManualSelected:
        class VideoChanged:
            endpoint = "/manual_selected_entity_changed"

            def __init__(
                self,
                dataset_id: int,
                team_id: int,
                workspace_id: int,
                project_id: int,
                figure_id: int,
                video_id: int,
                frame: int,
                tool_class_id: int,
                session_id: str,
                tool: str,
                user_id: int,
                job_id: int,
            ):
                self.dataset_id = dataset_id
                self.team_id = team_id
                self.workspace_id = workspace_id
                self.project_id = project_id
                self.figure_id = figure_id
                self.video_id = video_id
                self.frame = frame
                self.tool_class_id = tool_class_id
                self.session_id = session_id
                self.tool = tool
                self.user_id = user_id
                self.job_id = job_id

            @classmethod
            def from_json(cls, data: Dict[str, Any]):
                return cls(
                    dataset_id=data.get(ApiField.DATASET_ID),
                    team_id=data.get(ApiField.TEAM_ID),
                    workspace_id=data.get(ApiField.WORKSPACE_ID),
                    project_id=data.get(ApiField.PROJECT_ID),
                    figure_id=data.get(ApiField.FIGURE_ID),
                    video_id=data.get(ApiField.ENTITY_ID),
                    frame=data.get(ApiField.FRAME),
                    tool_class_id=data.get(ApiField.TOOL_CLASS_ID),
                    session_id=data.get(ApiField.SESSION_ID),
                    tool=data.get(ApiField.LABELING_TOOL),
                    user_id=data.get(ApiField.USER_ID),
                    job_id=data.get(ApiField.JOB_ID),
                )


def create(
    process_id=None,
    headless=False,
    auto_widget_id=False,
    before_shutdown_callbacks=None,
) -> FastAPI:
    from supervisely.app import DataJson, StateJson

    JinjaWidgets().auto_widget_id = auto_widget_id
    logger.info(f"JinjaWidgets().auto_widget_id is set to {auto_widget_id}.")

    app = FastAPI()
    WebsocketManager().set_app(app)

    @app.post("/shutdown")
    async def shutdown_endpoint(request: Request):
        shutdown(process_id, before_shutdown_callbacks)

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
            if (gettrace is not None and gettrace()) or is_development():
                response.headers["x-debug-mode"] = "1"
            return response

        @app.post("/session-info")
        async def send_session_info(request: Request):
            # TODO: handle case development inside docker
            production_at_instance = is_production() and is_docker()
            advanced_debug = is_debug_with_sly_net()
            development = is_development() or (is_production() and not is_docker())

            if advanced_debug or development:
                server_address = sly_env.server_address(raise_not_found=False)
                if server_address is not None:
                    server_address = Api.normalize_server_address(server_address)
            elif production_at_instance:
                server_address = "/"
            else:
                raise ValueError(
                    "'Unrecognized running mode, should be one of ['advanced_debug', 'development', 'production']."
                )

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

        app.mount("/css", StaticFiles(directory=supervisely.__path__[0]), name="sly_static")

    return app


def shutdown(
    process_id=None,
    before_shutdown_callbacks: Optional[List[Callable[[], None]]] = None,
):
    logger.info(f"Shutting down [pid argument = {process_id}]...")

    if before_shutdown_callbacks is not None:
        logger.info("Found tasks to run before shutdown.")
        for func in before_shutdown_callbacks:
            logger.debug(f"Call {func.__name__}")
            func()
    else:
        logger.debug("No tasks to call before shutdown")

    try:
        if process_id is None:
            # process_id = psutil.Process(os.getpid()).ppid()
            process_id = os.getpid()
        current_process = psutil.Process(process_id)
        if os.name == "nt":
            # for windows
            current_process.send_signal(signal.CTRL_C_EVENT)  # emit ctrl + c
        else:
            current_process.send_signal(signal.SIGINT)  # emit ctrl + c
    except KeyboardInterrupt:
        logger.info("Application has been shut down successfully")


def enable_hot_reload_on_debug(app: FastAPI):
    templates = Jinja2Templates()
    gettrace = getattr(sys, "gettrace", None)
    if gettrace is None:
        print("Can not detect debug mode, no sys.gettrace")
    elif gettrace():
        # List of directories to exclude from the hot reload.
        exclude = [".venv", ".git", "tmp"]

        hot_reload = arel.HotReload(
            paths=[arel.Path(path) for path in os.listdir() if path not in exclude]
        )

        app.add_websocket_route("/hot-reload", route=hot_reload, name="hot-reload")
        app.add_event_handler("startup", hot_reload.startup)
        app.add_event_handler("shutdown", hot_reload.shutdown)
        templates.env.globals["HOTRELOAD"] = "1"
        templates.env.globals["hot_reload"] = hot_reload
        logger.debug("Debugger (gettrace) detected, UI hot-reload is enabled")
    else:
        logger.debug("In runtime mode ...")


async def process_server_error(request, exc: Exception, need_to_handle_error=True):
    from supervisely.io.exception_handlers import handle_exception

    handled_exception = handle_exception(exc) if need_to_handle_error else None

    if handled_exception is not None:
        details = {"title": handled_exception.title, "message": handled_exception.message}
        log_message = handled_exception.get_message_for_modal_window()
    else:
        details = {"title": "Oops! Something went wrong", "message": repr(exc)}
        log_message = repr(exc)
    if isinstance(exc, DialogWindowBase):
        details["title"] = exc.title
        details["message"] = exc.description
        details["status"] = exc.status

    if is_production():
        logger.error(
            log_message,
            exc_info=True,
            extra={"main_name": "main", "exc_str": details["message"]},
        )
    else:
        raise exc

    return await http_exception_handler(
        request,
        HTTPException(status_code=500, detail=details),
    )


def handle_server_errors(app: FastAPI):
    @app.exception_handler(500)
    async def server_exception_handler(request, exc):
        return await process_server_error(request, exc)


def _init(
    app: FastAPI = None,
    templates_dir: str = "templates",
    headless=False,
    process_id=None,
    static_dir=None,
    hot_reload=False,
    before_shutdown_callbacks=None,
) -> FastAPI:
    from supervisely.app.content import DataJson, StateJson
    from supervisely.app.fastapi import available_after_shutdown

    if app is None:
        app = _MainServer().get_server()

    handle_server_errors(app)

    if headless is False:
        if "app_body_padding" not in StateJson():
            StateJson()["app_body_padding"] = "20px"
        Jinja2Templates(directory=[str(Path(__file__).parent.absolute()), templates_dir])
        if hot_reload:
            enable_hot_reload_on_debug(app)

    StateJson()["slyAppShowDialog"] = False
    DataJson()["slyAppDialogTitle"] = ""
    DataJson()["slyAppDialogMessage"] = ""

    app.mount(
        "/sly",
        create(
            process_id,
            headless,
            auto_widget_id=True,
            before_shutdown_callbacks=before_shutdown_callbacks,
        ),
    )

    @app.middleware("http")
    async def get_state_from_request(request: Request, call_next):
        if headless is False:
            await StateJson.from_request(request)

        if not ("application/json" not in request.headers.get("Content-Type", "")):
            # {'command': 'inference_batch_ids', 'context': {}, 'state': {'dataset_id': 49711, 'batch_ids': [3120204], 'settings': None}, 'user_api_key': 'XXX', 'api_token': 'XXX', 'instance_type': None, 'server_address': 'https://dev.supervise.ly'}
            content = await request.json()

            request.state.context = content.get("context")
            request.state.state = content.get("state")
            request.state.api_token = content.get(
                "api_token",
                request.state.context.get("apiToken")
                if request.state.context is not None
                else None,
            )
            # logger.debug(f"middleware request api_token {request.state.api_token}")
            # request.state.server_address = content.get(
            #     "server_address", sly_env.server_address(raise_not_found=False)
            # )
            request.state.server_address = sly_env.server_address(raise_not_found=False)
            # logger.debug(f"middleware request server_address {request.state.server_address}")
            # logger.debug(f"middleware request context {request.state.context}")
            # logger.debug(f"middleware request state {request.state.state}")
            if request.state.server_address is not None and request.state.api_token is not None:
                request.state.api = Api(request.state.server_address, request.state.api_token)
            else:
                request.state.api = None

        try:
            response = await call_next(request)
        except Exception as exc:
            need_to_handle_error = is_production()
            response = await process_server_error(request, exc, need_to_handle_error)
        return response

    if headless is False:

        @app.get("/")
        @available_after_shutdown(app)
        def read_index(request: Request):
            return Jinja2Templates().TemplateResponse("index.html", {"request": request})

        @app.on_event("shutdown")
        def shutdown():
            from supervisely.app.content import ContentOrigin

            ContentOrigin().stop()
            client = TestClient(app)
            resp = run_sync(client.get("/"))
            assert resp.status_code == 200
            logger.info("Application has been shut down successfully")

        if static_dir is not None:
            app.mount("/static", CustomStaticFiles(directory=static_dir), name="static_files")

    return app


class _MainServer(metaclass=Singleton):
    def __init__(self):
        self._server = FastAPI()

    def get_server(self) -> FastAPI:
        return self._server


class Application(metaclass=Singleton):
    class StopException(Exception):
        """Raise to stop the function from running in app.handle_stop"""

    def __init__(
        self,
        layout: "Widget" = None,
        templates_dir: str = None,
        static_dir: str = None,
        hot_reload: bool = False,  # whether to use hot reload during debug or not (has no effect in production)
        session_info_extra_content: "Widget" = None,
        session_info_solid: bool = False,
    ):
        self._favicon = os.environ.get("icon", "https://cdn.supervise.ly/favicon.ico")
        JinjaWidgets().context["__favicon__"] = self._favicon
        JinjaWidgets().context["__no_html_mode__"] = True

        self._static_dir = static_dir

        self._stop_event = ThreadingEvent()
        # for backward compatibility
        self._graceful_stop_event: Optional[ThreadingEvent] = None

        def set_stop_event():
            self._stop_event.set()

        def wait_for_graceful_stop_event():
            if self._graceful_stop_event is None:
                return
            print_info = True
            while not self._graceful_stop_event.is_set():
                if print_info:
                    logger.info("Graceful shutdown. Waiting for `app.stop()` to be called.")
                    print_info = False
                sleep(0.1)

        self._before_shutdown_callbacks = [set_stop_event, wait_for_graceful_stop_event]

        headless = False
        if layout is None and templates_dir is None:
            templates_dir: str = "templates"  # for back compatibility
            headless = True
            logger.info(
                "Both arguments 'layout' and 'templates_dir' are not defined. App is headless (i.e. without UI)",
                extra={"templates_dir": templates_dir},
            )
        if layout is not None and templates_dir is not None:
            raise ValueError(
                "Only one of the arguments has to be defined: 'layout' or 'templates_dir'. 'layout' argument is recommended."
            )
        if layout is not None:
            templates_dir = os.path.join(Path(__file__).parent.absolute(), "templates")
            from supervisely.app.widgets import Identity

            main_layout = Identity(layout, widget_id="__app_main_layout__")
            logger.info(
                "Application is running in no-html mode", extra={"templates_dir": templates_dir}
            )
        else:
            JinjaWidgets().auto_widget_id = False
            JinjaWidgets().context["__no_html_mode__"] = False
        if session_info_extra_content is not None:
            session_info_extra_content_layout = Identity(
                session_info_extra_content, widget_id="__app_session_info_extra_content__"
            )

        if session_info_solid:
            JinjaWidgets().context["__app_session_info_solid__"] = True

        if is_production():
            logger.info("Application is running on Supervisely Platform in production mode")
        else:
            logger.info("Application is running on localhost in development mode")
        self._process_id = os.getpid()
        logger.info(f"Application PID is {self._process_id}")
        self._fastapi: FastAPI = _init(
            app=None,
            templates_dir=templates_dir,
            headless=headless,
            process_id=self._process_id,
            static_dir=static_dir,
            hot_reload=hot_reload,
            before_shutdown_callbacks=self._before_shutdown_callbacks,
        )
        self.test_client = TestClient(self._fastapi)

        if not headless:
            templates = Jinja2Templates()
            self.hot_reload = arel.HotReload([])
            self._fastapi.add_websocket_route(
                "/hot-reload", route=self.hot_reload, name="hot-reload"
            )
            self._fastapi.add_event_handler("startup", self.hot_reload.startup)
            self._fastapi.add_event_handler("shutdown", self.hot_reload.shutdown)

            # Setting HOTRELOAD=1 in template context, otherwise the HTML would not have the hot reload script.
            templates.env.globals["HOTRELOAD"] = "1"
            templates.env.globals["hot_reload"] = self.hot_reload

            logger.debug("Hot reload is enabled, use app.reload_page() to reload page.")

            if is_production():
                # to save offline session
                from supervisely.app.content import ContentOrigin

                ContentOrigin().start()
                resp = run_sync(self.test_client.get("/"))

    def get_server(self):
        return self._fastapi

    async def __call__(self, scope, receive, send) -> None:
        await self._fastapi.__call__(scope, receive, send)

    def shutdown(self):
        shutdown(self._process_id, self._before_shutdown_callbacks)

    def stop(self):
        if self._graceful_stop_event is not None:
            self._graceful_stop_event.set()
        if self.is_stopped():
            return
        self.shutdown()

    def is_stopped(self):
        """Indicates whether the application is in the process of being terminated."""
        return self._stop_event.is_set()

    def reload_page(self):
        run_sync(self.hot_reload.notify.notify())

    def get_static_dir(self):
        return self._static_dir

    def call_before_shutdown(self, func: Callable[[], None]):
        self._before_shutdown_callbacks.append(func)

    def handle_stop(self, graceful: bool = True):
        """Contextmanager to suppress StopException and control graceful shutdown.

        :param graceful: Whether to perform a graceful shutdown if a StopException is raised.
        If set to `False` and shutdown request recieved (i.e. `app.is_stopped()` is `True`),
        the application will be terminated immediately, defaults to `True`
        :type graceful: bool
        :return: context manager
        :rtype: _type_
        """
        self._graceful_stop_event = ThreadingEvent()
        if graceful is False:
            self._graceful_stop_event.set()
        return suppress(self.StopException)

    def event(self, event: Event) -> Callable:
        """Decorator to register posts to specific endpoints.

        :param event: event to register (e.g. `Event.Brush.LeftMouseReleased`)
        :type event: Event
        :return: decorator
        :rtype: Callable

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            app = sly.Application(layout=layout)

            @app.event(sly.Event.Brush.LeftMouseReleased)
            def some_function(api: sly.Api, event: sly.Event.Brush.LeftMouseReleased):
                # do something
                pass
        """

        def inner(func: Callable) -> Callable:
            server = self.get_server()

            @server.post(event.endpoint)
            def wrapper(request: Request):
                return func(request.state.api, event.from_json(request.state.context))

            return wrapper

        return inner


def set_autostart_flag_from_state(default: Optional[str] = None):
    """Set `autostart` flag recieved from task state. Env name: `modal.state.autostart`.

    :param default: this value will be set
        if the flag is undefined in state, defaults to None
    :type default: Optional[str], optional
    """
    if sly_env.autostart() is True:
        logger.info("`autostart` flag already defined in env. Skip loading it from state.")
        return

    api = Api()
    task_id = sly_env.task_id(raise_not_found=False)
    if task_id is None:
        logger.warn("`autostart` env can't be setted: TASK_ID variable is not defined.")
        return
    task_meta = api.task.get_info_by_id(task_id).get("meta", None)
    task_params = None
    task_state = None
    auto_start = default
    if task_meta is not None:
        task_params = task_meta.get("params", None)
    if task_params is not None:
        task_state = task_params.get("state", None)
    if task_state is not None:
        auto_start = task_params.get("autostart", default)

    sly_env.set_autostart(auto_start)


def call_on_autostart(
    default_func: Optional[Callable] = None,
    **default_kwargs,
) -> Callable:
    """Decorator to enable autostart.
    This decorator is used to wrap functions that are executed
    and will check if autostart is enabled in environment.

    :param default_func: default function to call if autostart is not enabled, defaults to None
    :type default_func: Optional[Callable], optional
    :return: decorator
    :rtype: Callable
    """
    set_autostart_flag_from_state()

    def inner(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if sly_env.autostart() is True:
                logger.info("Found `autostart` flag in environment.")
                func(*args, **kwargs)
            else:
                logger.info("Autostart is disabled.")
                if default_func is not None:
                    default_func(**default_kwargs)

        return wrapper

    return inner


def get_name_from_env(default="Supervisely App"):
    return os.environ.get("APP_NAME", default)
