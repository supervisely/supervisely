import os
import signal
import psutil

# https://fastapi.tiangolo.com/advanced/sub-applications/
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from supervisely.fastapi_helpers import DataJson, LastStateJson
from supervisely.fastapi_helpers import WebsocketManager
from supervisely.io.fs import mkdir, dir_exists
from supervisely import logger


def create_supervisely_app() -> FastAPI:
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
    current_process.send_signal(signal.SIGINT) # emit ctrl + c


def get_app_data_dir():
    key = 'SLY_APP_DATA_DIR'
    dir = None
    
    try:
        dir = os.environ[key]
    except KeyError as e:
        raise KeyError(f"Environment variable {key} is not defined")
    
    if dir_exists(dir) is False:
        logger.warn(f"App data directory {dir} doesn't exist. Will be made automatically.")
        mkdir(dir)
    
    return dir
