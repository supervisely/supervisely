# https://fastapi.tiangolo.com/advanced/websockets/#handling-disconnections-and-multiple-clients
# https://github.com/tiangolo/fastapi/issues/2639
# https://github.com/tiangolo/fastapi/issues/1501#issuecomment-638219871

from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


class WebsocketManager:
    
    def __init__(self, app: FastAPI, path="/sly-app-ws"):
        self._app=app
        self.path=path
        self.active_connections: List[WebSocket] = []
        if not hasattr(self._app.state, 'SLY_WS'):
            setattr(self._ap.state, "SLY_WS", True)
        
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

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
