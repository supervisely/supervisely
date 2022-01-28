from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi.responses import JSONResponse
from supervisely.fastapi_helpers.app_content import DataJson


class DataMiddleware:
    def __init__(
        self, 
        app: ASGIApp,
        path = "/sly-app-data"
    ) -> None:
        self.app = app
        self.path = path
        
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] == "GET":
            return await self.app(scope, receive, send)        
        if scope["path"] == self.path:
            actual_data = DataJson()
            response = JSONResponse(content={actual_data._field: dict(actual_data)})
            return await response(scope, receive, send) 
        
        return await self.app(scope, receive, send)
