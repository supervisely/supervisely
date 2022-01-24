# https://fastapi.tiangolo.com/advanced/websockets/#handling-disconnections-and-multiple-clients
# https://github.com/tiangolo/fastapi/issues/2639
# https://github.com/tiangolo/fastapi/issues/1501#issuecomment-638219871

import os
import signal
import psutil
from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi import Response
from typing import List
from fastapi import WebSocket, WebSocketDisconnect


class WebsocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, state: dict): 
        for connection in self.active_connections:
            await connection.send_json(state)

    async def endpoint(self, websocket: WebSocket):
        await self.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
        except WebSocketDisconnect:
            self.disconnect(websocket)


class WebsocketMiddleware:
    def __init__(
        self, app: ASGIApp, ws_manager: WebsocketManager, path: str = "/ws"
    ) -> None:
        self.app = app
        self.ws_manager = ws_manager
        self.path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "ws" and scope["path"] == self.path:
            print(123)
            # if scope["path"] == self.path:
            #     scope["app"].state.STOPPED = True
            #     self.shutdown()
            # if hasattr(scope["app"].state, 'STOPPED'):
            #     stopped = scope["app"].state.STOPPED
            #     if stopped:
            #         # PlainTextResponse("Invalid host header", status_code=400)
            #         response = Response(content="Server is being shut down", status_code=403)
            #         return await response(scope, receive, send) 
        
        await self.app(scope, receive, send)
