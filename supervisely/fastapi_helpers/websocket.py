from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from supervisely.fastapi_helpers.singleton import Singleton


class WebsocketManager(metaclass=Singleton):
    def __init__(self, app: FastAPI, path="/sly-app-ws"):
        self.app = app
        self.path = path
        self.active_connections: List[WebSocket] = []
        app.add_api_websocket_route(path=self.path, endpoint=self.endpoint)
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, d: dict): 
        for connection in self.active_connections:
            await connection.send_json(d)

    async def endpoint(self, websocket: WebSocket):
        await self.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
        except WebSocketDisconnect:
            self.disconnect(websocket)
