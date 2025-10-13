import hashlib
import time
from typing import Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import supervisely.io.env as sly_env
from supervisely.app.singleton import Singleton


class WebsocketManager(metaclass=Singleton):
    def __init__(self, path="/sly-app-ws"):
        self.app = None
        self.path = path
        self.active_connections: List[WebSocket] = []
        self._connection_users: Dict[WebSocket, Optional[Union[int, str]]] = {}
        self._cookie_user_map: Dict[str, Tuple[Union[int, str], float]] = {}
        self._cookie_ttl_seconds = 60 * 60

    def set_app(self, app: FastAPI):
        if self.app is not None:
            return
        self.app = app
        self.app.add_api_websocket_route(path=self.path, endpoint=self.endpoint)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        user_id = self._resolve_user_id(websocket)
        self.active_connections.append(websocket)
        self._connection_users[websocket] = user_id

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self._connection_users.pop(websocket, None)

    def remember_user_cookie(
        self, cookie_header: Optional[str], user_id: Optional[Union[int, str]]
    ):
        if cookie_header is None or user_id is None:
            return
        fingerprint = self._cookie_fingerprint(cookie_header)
        if fingerprint is None:
            return
        self._purge_cookie_cache()
        self._cookie_user_map[fingerprint] = (user_id, time.monotonic())

    async def broadcast(self, d: dict, user_id: Optional[Union[int, str]] = None):
        if sly_env.is_multiuser_mode_enabled():
            if user_id is None:
                user_id = sly_env.user_from_multiuser_app()
            if user_id is None:
                targets = list(self.active_connections)
            else:
                targets = [
                    connection
                    for connection in self.active_connections
                    if self._connection_users.get(connection) == user_id
                ]
        else:
            targets = list(self.active_connections)

        for connection in list(targets):
            await connection.send_json(d)

    async def endpoint(self, websocket: WebSocket):
        await self.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
        except WebSocketDisconnect:
            self.disconnect(websocket)

    def _resolve_user_id(self, websocket: WebSocket) -> Optional[int]:
        if not sly_env.is_multiuser_mode_enabled():
            return None
        query_user = websocket.query_params.get("userId")
        if query_user is not None:
            try:
                return int(query_user)
            except ValueError:
                pass
        fingerprint = self._cookie_fingerprint(websocket.headers.get("cookie"))
        if fingerprint is None:
            return None
        cached = self._cookie_user_map.get(fingerprint)
        if cached is None:
            return None
        user_id, ts = cached
        if time.monotonic() - ts > self._cookie_ttl_seconds:
            self._cookie_user_map.pop(fingerprint, None)
            return None
        return user_id

    @staticmethod
    def _cookie_fingerprint(cookie_header: Optional[str]) -> Optional[str]:
        if not cookie_header:
            return None
        return hashlib.sha256(cookie_header.encode("utf-8")).hexdigest()

    def _purge_cookie_cache(self) -> None:
        if not self._cookie_user_map:
            return
        cutoff = time.monotonic() - self._cookie_ttl_seconds
        expired = [key for key, (_, ts) in self._cookie_user_map.items() if ts < cutoff]
        for key in expired:
            self._cookie_user_map.pop(key, None)
