from urllib.request import Request
from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi.responses import JSONResponse


class DataMiddleware:
    def __init__(
        self, 
        app: ASGIApp,
        data: dict,
        path = "/sly-app-data"
    ) -> None:
        self.app = app
        self.data = data
        self.path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http": 
            if scope["path"] == self.path:
                response = JSONResponse(content=self.data)
                return await response(scope, receive, send) 
        await self.app(scope, receive, send)
