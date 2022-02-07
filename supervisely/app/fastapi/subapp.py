import os
import signal
import psutil

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# import supervisely as sly

from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.io.fs import mkdir, dir_exists
from supervisely.sly_logger import logger


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



