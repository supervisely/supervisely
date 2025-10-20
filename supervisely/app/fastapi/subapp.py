import hashlib
import inspect
import json
import os
import signal
import sys
import time
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from threading import Event as ThreadingEvent
from threading import Thread
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import arel
import jinja2
import numpy as np
import psutil
from async_asgi_testclient import TestClient
from cachetools import TTLCache
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
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles

import supervisely.app.fastapi.multi_user as multi_user
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
from supervisely.sly_logger import create_formatter, logger

# from supervisely.app.fastapi.request import Request

if TYPE_CHECKING:
    from supervisely.app.widgets import Widget

import logging

SUPERVISELY_SERVER_PATH_PREFIX = sly_env.supervisely_server_path_prefix()
if SUPERVISELY_SERVER_PATH_PREFIX and not SUPERVISELY_SERVER_PATH_PREFIX.startswith("/"):
    SUPERVISELY_SERVER_PATH_PREFIX = f"/{SUPERVISELY_SERVER_PATH_PREFIX}"

HEALTH_ENDPOINTS = ["/health", "/is_ready"]

# Context variable for response time
response_time_ctx: ContextVar[float] = ContextVar("response_time", default=None)

# Mapping from user_id to Api instance
_USER_API_CACHE = TTLCache(maxsize=500, ttl=60 * 15)  # Cache up to 15 minutes


class ReadyzFilter(logging.Filter):
    def filter(self, record):
        if "/readyz" in record.getMessage() or "/livez" in record.getMessage():
            record.levelno = logging.DEBUG  # Change log level to DEBUG
            record.levelname = "DEBUG"
        return True


class ResponseTimeFilter(logging.Filter):
    def filter(self, record):
        # Check if this is an HTTP access log line by logger name
        if getattr(record, "name", "") == "uvicorn.access":
            response_time = response_time_ctx.get(None)
            if response_time is not None:
                record.responseTime = int(response_time)
        return True


def _init_uvicorn_logger():
    uvicorn_logger = logging.getLogger("uvicorn.access")
    for handler in uvicorn_logger.handlers:
        handler.setFormatter(create_formatter())
    uvicorn_logger.addFilter(ReadyzFilter())
    uvicorn_logger.addFilter(ResponseTimeFilter())


_init_uvicorn_logger()


class PrefixRouter(APIRouter):
    def add_api_route(self, path, *args, **kwargs):
        allowed_paths = ["/livez", "/is_alive", "/is_running", "/readyz", "/is_ready"]
        if path in allowed_paths:
            super().add_api_route(path, *args, **kwargs)
        if SUPERVISELY_SERVER_PATH_PREFIX:
            path = SUPERVISELY_SERVER_PATH_PREFIX + path
        super().add_api_route(path, *args, **kwargs)


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
                        raise ValueError(f"Unknown tool option: {tool_option}")
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

        class FigureChanged:
            endpoint = "/manual_selected_figure_changed"

            def __init__(
                self,
                dataset_id: int,
                team_id: int,
                workspace_id: int,
                project_id: int,
                figure_id: int,
                figure_class_id: int,
                figure_class_title: str,
                image_id: int,
                video_id: int,
                frame: int,
                object_id: int,
                tool_class_id: int,
                session_id: str,
                tool: str,
                user_id: int,
                job_id: int,
                previous_figure: dict = None,
            ):
                self.dataset_id = dataset_id
                self.team_id = team_id
                self.workspace_id = workspace_id
                self.project_id = project_id
                self.figure_id = figure_id
                self.image_id = image_id
                self.figure_class_id = figure_class_id
                self.figure_class_title = figure_class_title
                self.video_id = video_id
                self.frame = frame
                self.object_id = object_id
                self.tool_class_id = tool_class_id
                self.session_id = session_id
                self.tool = tool
                self.user_id = user_id
                self.job_id = job_id
                self.previous_figure = previous_figure

            @classmethod
            def from_json(cls, data: dict):
                return cls(
                    dataset_id=data.get(ApiField.DATASET_ID),
                    team_id=data.get(ApiField.TEAM_ID),
                    workspace_id=data.get(ApiField.WORKSPACE_ID),
                    project_id=data.get(ApiField.PROJECT_ID),
                    figure_id=data.get(ApiField.FIGURE_ID),
                    figure_class_id=data.get(ApiField.FIGURE_CLASS_ID),
                    figure_class_title=data.get(ApiField.FIGURE_CLASS_TITLE),
                    image_id=data.get(ApiField.ENTITY_ID),
                    video_id=data.get(ApiField.ENTITY_ID),
                    frame=data.get(ApiField.FRAME),
                    object_id=data.get(ApiField.ANNOTATION_OBJECT_ID),
                    tool_class_id=data.get(ApiField.TOOL_CLASS_ID),
                    session_id=data.get(ApiField.SESSION_ID),
                    tool=data.get(ApiField.LABELING_TOOL),
                    user_id=data.get(ApiField.USER_ID),
                    job_id=data.get(ApiField.JOB_ID),
                    previous_figure=data.get(
                        "previousFigure", None
                    ),  # there is no such field in ApiField
                )

        class ImageChanged:
            """This event is triggered when the user changes the image in the Image Labeling Tooolbox."""

            endpoint = "/manual_selected_image_changed"

            def __init__(
                self,
                dataset_id: int,
                team_id: int,
                workspace_id: int,
                project_id: int,
                image_id: int,
                figure_id: int,
                figure_class_id: int,
                figure_class_title: str,
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
                self.image_id = image_id
                self.figure_id = figure_id
                self.figure_class_id = figure_class_id
                self.figure_class_title = figure_class_title
                self.tool_class_id = tool_class_id
                self.session_id = session_id
                self.tool = tool
                self.user_id = user_id
                self.job_id = job_id

            @classmethod
            def from_json(cls, data: dict):
                return cls(
                    dataset_id=data.get(ApiField.DATASET_ID),
                    team_id=data.get(ApiField.TEAM_ID),
                    workspace_id=data.get(ApiField.WORKSPACE_ID),
                    project_id=data.get(ApiField.PROJECT_ID),
                    image_id=data.get(ApiField.IMAGE_ID),
                    figure_id=data.get(ApiField.FIGURE_ID),
                    figure_class_id=data.get(ApiField.FIGURE_CLASS_ID),
                    figure_class_title=data.get(ApiField.FIGURE_CLASS_TITLE),
                    tool_class_id=data.get(ApiField.TOOL_CLASS_ID),
                    session_id=data.get(ApiField.SESSION_ID),
                    tool=data.get(ApiField.LABELING_TOOL),
                    user_id=data.get(ApiField.USER_ID),
                    job_id=data.get(ApiField.JOB_ID),
                )

    class FigureCreated:
        endpoint = "/figure_created"

        def __init__(
            self,
            dataset_id: int,
            team_id: int,
            workspace_id: int,
            project_id: int,
            figure_id: int,
            figure_class_id: int,
            figure_class_title: str,
            image_id: int,
            video_id: int,
            frame: int,
            object_id: int,
            tool_class_id: int,
            session_id: str,
            tool: str,
            user_id: int,
            job_id: int,
            tool_state: dict,
            figure_state: dict,
        ):
            self.dataset_id = dataset_id
            self.team_id = team_id
            self.workspace_id = workspace_id
            self.project_id = project_id
            self.figure_id = figure_id
            self.figure_class_id = figure_class_id
            self.figure_class_title = figure_class_title
            self.image_id = image_id
            self.video_id = video_id
            self.frame = frame
            self.object_id = object_id
            self.tool_class_id = tool_class_id
            self.session_id = session_id
            self.tool = tool
            self.user_id = user_id
            self.job_id = job_id
            self.tool_state = tool_state
            self.figure_state = figure_state

        @classmethod
        def from_json(cls, data: dict):
            return cls(
                dataset_id=data.get(ApiField.DATASET_ID),
                team_id=data.get(ApiField.TEAM_ID),
                workspace_id=data.get(ApiField.WORKSPACE_ID),
                project_id=data.get(ApiField.PROJECT_ID),
                figure_id=data.get(ApiField.FIGURE_ID),
                figure_class_id=data.get(ApiField.FIGURE_CLASS_ID),
                figure_class_title=data.get(ApiField.FIGURE_CLASS_TITLE),
                image_id=data.get(ApiField.IMAGE_ID),
                video_id=data.get(ApiField.ENTITY_ID),
                frame=data.get(ApiField.FRAME),
                object_id=data.get(ApiField.ANNOTATION_OBJECT_ID),
                tool_class_id=data.get(ApiField.TOOL_CLASS_ID),
                session_id=data.get(ApiField.SESSION_ID),
                tool=data.get(ApiField.LABELING_TOOL),
                user_id=data.get(ApiField.USER_ID),
                job_id=data.get(ApiField.JOB_ID),
                tool_state=data.get(ApiField.JOB_ID),
                figure_state=data.get(ApiField.FIGURE_STATE),
            )

    class Tools:
        class Rectangle:
            class FigureChanged:
                endpoint = "/tools_rectangle_figure_changed"

                def __init__(
                    self,
                    dataset_id: int,
                    team_id: int,
                    workspace_id: int,
                    project_id: int,
                    figure_id: int,
                    figure_class_id: int,
                    figure_class_title: str,
                    image_id: int,
                    tool_class_id: int,
                    session_id: str,
                    tool: str,
                    user_id: int,
                    job_id: int,
                    tool_state: dict,
                    figure_state: dict,
                ):
                    self.dataset_id = dataset_id
                    self.team_id = team_id
                    self.workspace_id = workspace_id
                    self.project_id = project_id
                    self.figure_id = figure_id
                    self.figure_class_id = figure_class_id
                    self.figure_class_title = figure_class_title
                    self.image_id = image_id
                    self.tool_class_id = tool_class_id
                    self.session_id = session_id
                    self.tool = tool
                    self.user_id = user_id
                    self.job_id = job_id
                    self.tool_state = tool_state
                    self.figure_state = figure_state

                @classmethod
                def from_json(cls, data: dict):
                    return cls(
                        dataset_id=data.get(ApiField.DATASET_ID),
                        team_id=data.get(ApiField.TEAM_ID),
                        workspace_id=data.get(ApiField.WORKSPACE_ID),
                        project_id=data.get(ApiField.PROJECT_ID),
                        figure_id=data.get(ApiField.FIGURE_ID),
                        figure_class_id=data.get(ApiField.FIGURE_CLASS_ID),
                        figure_class_title=data.get(ApiField.FIGURE_CLASS_TITLE),
                        image_id=data.get(ApiField.IMAGE_ID),
                        tool_class_id=data.get(ApiField.TOOL_CLASS_ID),
                        session_id=data.get(ApiField.SESSION_ID),
                        tool=data.get(ApiField.LABELING_TOOL),
                        user_id=data.get(ApiField.USER_ID),
                        job_id=data.get(ApiField.JOB_ID),
                        tool_state=data.get(ApiField.JOB_ID),
                        figure_state=data.get(ApiField.FIGURE_STATE),
                    )

    class Entity:
        class FrameChanged:
            endpoint = "/entity_frame_changed"

            def __init__(
                self,
                dataset_id: int,
                team_id: int,
                workspace_id: int,
                project_id: int,
                figure_id: int,
                figure_class_id: int,
                figure_class_title: str,
                video_id: int,
                frame: int,
                object_id: int,
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
                self.figure_class_id = figure_class_id
                self.figure_class_title = figure_class_title
                self.video_id = video_id
                self.frame = frame
                self.object_id = object_id
                self.tool_class_id = tool_class_id
                self.session_id = session_id
                self.tool = tool
                self.user_id = user_id
                self.job_id = job_id

            @classmethod
            def from_json(cls, data: dict):
                return cls(
                    dataset_id=data.get(ApiField.DATASET_ID),
                    team_id=data.get(ApiField.TEAM_ID),
                    workspace_id=data.get(ApiField.WORKSPACE_ID),
                    project_id=data.get(ApiField.PROJECT_ID),
                    figure_id=data.get(ApiField.FIGURE_ID),
                    figure_class_id=data.get(ApiField.FIGURE_CLASS_ID),
                    figure_class_title=data.get(ApiField.FIGURE_CLASS_TITLE),
                    video_id=data.get(ApiField.ENTITY_ID),
                    frame=data.get(ApiField.FRAME),
                    object_id=data.get(ApiField.ANNOTATION_OBJECT_ID),
                    tool_class_id=data.get(ApiField.TOOL_CLASS_ID),
                    session_id=data.get(ApiField.SESSION_ID),
                    tool=data.get(ApiField.LABELING_TOOL),
                    user_id=data.get(ApiField.USER_ID),
                    job_id=data.get(ApiField.JOB_ID),
                )

    class JobEntity:
        class StatusChanged:
            endpoint = "/job_entity_status_changed"

            def __init__(
                self,
                dataset_id: int,
                team_id: int,
                workspace_id: int,
                project_id: int,
                figure_id: int,
                figure_class_id: int,
                figure_class_title: str,
                image_id: int,
                entity_id: int,
                tool_class_id: int,
                session_id: str,
                tool: str,
                user_id: int,
                job_id: int,
                job_entity_status: str,
            ):
                self.dataset_id = dataset_id
                self.team_id = team_id
                self.workspace_id = workspace_id
                self.project_id = project_id
                self.figure_id = figure_id
                self.figure_class_id = figure_class_id
                self.figure_class_title = figure_class_title
                self.image_id = image_id
                self.entity_id = entity_id
                self.tool_class_id = tool_class_id
                self.session_id = session_id
                self.tool = tool
                self.user_id = user_id
                self.job_id = job_id
                self.job_entity_status = job_entity_status

            @classmethod
            def from_json(cls, data: dict):
                return cls(
                    dataset_id=data.get(ApiField.DATASET_ID),
                    team_id=data.get(ApiField.TEAM_ID),
                    workspace_id=data.get(ApiField.WORKSPACE_ID),
                    project_id=data.get(ApiField.PROJECT_ID),
                    figure_id=data.get(ApiField.FIGURE_ID),
                    figure_class_id=data.get(ApiField.FIGURE_CLASS_ID),
                    figure_class_title=data.get(ApiField.FIGURE_CLASS_TITLE),
                    image_id=data.get(ApiField.IMAGE_ID),
                    entity_id=data.get(ApiField.ENTITY_ID),
                    tool_class_id=data.get(ApiField.TOOL_CLASS_ID),
                    session_id=data.get(ApiField.SESSION_ID),
                    tool=data.get(ApiField.LABELING_TOOL),
                    user_id=data.get(ApiField.USER_ID),
                    job_id=data.get(ApiField.JOB_ID),
                    job_entity_status=data.get(ApiField.JOB_ENTITY_STATUS),
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
            if not sly_env.is_multiuser_mode_enabled():
                data = DataJson()
                response = JSONResponse(content=dict(data))
                return response
            user_id = await multi_user.extract_user_id_from_request(request)
            multi_user.remember_cookie(request, user_id)
            with multi_user.session_context(user_id):
                data = DataJson()
                response = JSONResponse(content=dict(data))
            return response

        @app.post("/state")
        async def send_state(request: Request):
            if not sly_env.is_multiuser_mode_enabled():
                state = StateJson()
                response = JSONResponse(content=dict(state))
            else:
                user_id = await multi_user.extract_user_id_from_request(request)
                multi_user.remember_cookie(request, user_id)
                with multi_user.session_context(user_id):
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
        # Start timer for response time measurement
        start_time = time.perf_counter()

        async def _process_request(request: Request, call_next):
            if "application/json" in request.headers.get("Content-Type", ""):
                content = await request.json()
                request.state.context = content.get("context")
                request.state.state = content.get("state")
                request.state.api_token = content.get(
                    "api_token",
                    (
                        request.state.context.get("apiToken")
                        if request.state.context is not None
                        else None
                    ),
                )
                request.state.server_address = content.get(
                    "server_address", sly_env.server_address(raise_not_found=False)
                )
                if (
                    request.state.server_address is not None
                    and request.state.api_token is not None
                ):
                    request.state.api = Api(
                        request.state.server_address, request.state.api_token
                    )
                    if sly_env.is_multiuser_mode_enabled():
                        user_id = sly_env.user_from_multiuser_app()
                        if user_id is not None:
                            _USER_API_CACHE[user_id] = request.state.api
                else:
                    request.state.api = None

            try:
                response = await call_next(request)
            except Exception as exc:
                need_to_handle_error = is_production()
                response = await process_server_error(
                    request, exc, need_to_handle_error
                )

            return response

        if not sly_env.is_multiuser_mode_enabled():
            if headless is False:
                await StateJson.from_request(request)
            response = await _process_request(request, call_next)
        else:
            user_id = await multi_user.extract_user_id_from_request(request)
            multi_user.remember_cookie(request, user_id)

            with multi_user.session_context(user_id):
                if headless is False:
                    await StateJson.from_request(request, local=False)
                response = await _process_request(request, call_next)
        # Calculate response time and set it for uvicorn logger in ms
        elapsed_ms = round((time.perf_counter() - start_time) * 1000)
        response_time_ctx.set(elapsed_ms)
        return response

    def verify_localhost(request: Request):
        client_host = request.client.host
        if client_host not in ["127.0.0.1", "::1"]:
            raise HTTPException(status_code=404, detail="Not Found")

    @app.post("/debug", dependencies=[Depends(verify_localhost)])
    def start_debug():
        import debugpy

        debug_host = os.getenv("DEBUG_HOST", "127.0.0.1")
        debug_port = int(os.getenv("DEBUG_PORT", "5678"))
        debugpy.listen((debug_host, debug_port))
        return {
            "status": "success",
            "message": f"Debug server is listening on {debug_host}:{debug_port}",
            "host": debug_host,
            "port": debug_port,
        }

    if headless is False:
        app.cached_template = None

        @app.get("/")
        @available_after_shutdown(app)
        def read_index(request: Request):
            if request.query_params.get("ping", False) in ("true", "True", True, 1, "1"):
                return JSONResponse(content={"message": "App is running"}, status_code=200)
            if app.cached_template is None:
                app.cached_template = Jinja2Templates().TemplateResponse(
                    "index.html", {"request": request}
                )
            return app.cached_template

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
        self._server.router = PrefixRouter()

    def get_server(self) -> FastAPI:
        return self._server


class Application(metaclass=Singleton):
    class StopException(Exception):
        """Raise to stop the function from running in app.handle_stop"""

    def __init__(
        self,
        layout: "Widget" = None,
        templates_dir: Optional[str] = None,
        static_dir: Optional[str] = None,
        hot_reload: bool = False,  # whether to use hot reload during debug or not (has no effect in production)
        session_info_extra_content: "Widget" = None,
        session_info_solid: bool = False,
        ready_check_function: Optional[
            Callable
        ] = None,  # function to check if the app is ready for requests (e.g serving app: model is served and ready)
        show_header: bool = True,
        hide_health_check_logs: bool = True,  # whether to hide health check logs in info level
        health_check_endpoints: Optional[List[str]] = None,  # endpoints to check health of the app
    ):
        """Initialize the Supervisely Application.

        :param layout: Main layout of the application.
        :type layout: Widget
        :param templates_dir: Directory with Jinja2 templates. It is preferred to use `layout` instead of `templates_dir`.
        :type templates_dir: str, optional
        :param static_dir: Directory with static files (e.g. CSS, JS), used for serving static content.
        :type static_dir: str, optional
        :param hot_reload: Whether to enable hot reload during development (default is False).
        :type hot_reload: bool, optional
        :param session_info_extra_content: Additional content to be displayed in the session info area.
        :type session_info_extra_content: Widget, optional
        :param session_info_solid: Whether to use solid background for the session info area.
        :type session_info_solid: bool, optional
        :param ready_check_function: Function to check if the app is ready for requests.
        :type ready_check_function: Callable, optional
        :param show_header: Whether to show the header in the application.
        :type show_header: bool, optional
        :param hide_health_check_logs: Whether to hide health check logs in info level.
        :type hide_health_check_logs: bool, optional
        :param health_check_endpoints: List of additional endpoints to check health of the app.
            Add your custom endpoints here to be able to manage logging of health check requests on info level with `hide_health_check_logs`.
        :type health_check_endpoints: List[str], optional
        """
        self._favicon = os.environ.get("icon", "https://cdn.supervisely.com/favicon.ico")
        JinjaWidgets().context["__favicon__"] = self._favicon
        JinjaWidgets().context["__no_html_mode__"] = True
        JinjaWidgets().context["__show_header__"] = show_header

        self._static_dir = static_dir

        self._stop_event = ThreadingEvent()
        # for backward compatibility
        self._graceful_stop_event: Optional[ThreadingEvent] = None
        self.set_ready_check_function(ready_check_function)

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

        # add filter to hide health check logs for info level
        if health_check_endpoints is None or len(health_check_endpoints) == 0:
            self._health_check_endpoints = HEALTH_ENDPOINTS
        else:
            health_check_endpoints = [endpoint.strip() for endpoint in health_check_endpoints]
            self._health_check_endpoints = HEALTH_ENDPOINTS + health_check_endpoints

        if hide_health_check_logs:
            self._setup_health_check_filter()

        self.test_client = TestClient(self._fastapi)

        if not headless:
            if is_development() and hot_reload:
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
                Thread(target=run_sync, args=(self.test_client.get("/"),)).start()

        server = self.get_server()

        @server.get("/livez")
        @server.get("/is_alive")
        @server.post("/is_running")
        async def is_running(request: Request):
            is_running = True
            if is_production():
                # @TODO: set task status to running
                return {"running": is_running, "mode": "production"}
            else:
                return {"running": is_running, "mode": "development"}

        @server.get("/readyz")
        @server.get("/is_ready")
        @server.post("/is_ready")
        async def is_ready(response: Response, request: Request):
            is_ready = True
            if self._ready_check_function is not None:
                is_ready = self._ready_check_function()
            if is_ready is False:
                raise HTTPException(status_code=503, detail="Service not ready")
            return {"status": "ready"}

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

    def event(self, event: Event, use_state: bool = False) -> Callable:
        """Decorator to register posts to specific endpoints.
        Supports both async and sync functions.

        :param event: event to register (e.g. `Event.Brush.LeftMouseReleased`)
        :type event: Event
        :param use_state: if set to True, data will be extracted from request.state.state,
            otherwise from request.state.context, defaults to False
        :type use_state: bool, optional
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

            if inspect.iscoroutinefunction(func):

                @server.post(event.endpoint)
                async def wrapper(request: Request):
                    data = request.state.state if use_state else request.state.context
                    return await func(request.state.api, event.from_json(data))

            else:

                @server.post(event.endpoint)
                def wrapper(request: Request):
                    data = request.state.state if use_state else request.state.context
                    return func(request.state.api, event.from_json(data))

            return wrapper

        return inner

    def render(self, context: Dict[str, Any]):
        return Jinja2Templates().render("index.html", {**context, "HOTRELOAD": False})

    def set_ready_check_function(self, func: Callable):
        self._ready_check_function = func

    def _setup_health_check_filter(self):
        """Setup filter to hide health check logs for info level."""

        class HealthCheckFilter(logging.Filter):
            def __init__(self, app_instance):
                super().__init__()
                self.app: Application = app_instance

            def filter(self, record):
                # Hide health check requests if NOT in debug mode
                if not self.app._fastapi.debug and hasattr(record, "getMessage"):
                    message = record.getMessage()
                    # Check if the message contains health check paths
                    if any(path in message for path in self.app._health_check_endpoints):
                        return False
                return True

        # Apply filter to uvicorn access logger
        health_filter = HealthCheckFilter(self)
        uvicorn_logger = logging.getLogger("uvicorn.access")

        # Remove old filters of this type, if any (for safety)
        uvicorn_logger.filters = [
            f for f in uvicorn_logger.filters if not isinstance(f, HealthCheckFilter)
        ]

        uvicorn_logger.addFilter(health_filter)


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

def session_user_api() -> Optional[Api]:
    """Returns the API instance for the current session user."""
    if not sly_env.is_multiuser_mode_enabled():
        return Api.from_env()
    user_id = sly_env.user_from_multiuser_app()
    if user_id is None:
        return None
    return _USER_API_CACHE.get(user_id, None)
