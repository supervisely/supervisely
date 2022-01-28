from urllib.request import Request
from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi import Request
from fastapi.responses import JSONResponse
from supervisely.fastapi_helpers.app_content import LastStateJson


class StateMiddleware:
    def __init__(
        self, 
        app: ASGIApp,
        path = "/sly-app-state"
    ) -> None:
        self.app = app
        self.path = path
        LastStateJson()
        
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] == "GET":
            return await self.app(scope, receive, send)
        
        #@TODO: error implementation here
        # request = Request(scope, receive, send)
        # await LastStateJson.replace(request)

        # request = Request(scope, receive, send)
        # await LastStateJson.replace(request)

        if scope["path"] == self.path:
            last_state = LastStateJson()
            response = JSONResponse(content={last_state._field: dict(last_state)})
            return await response(scope, receive, send) 
        
        return await self.app(scope, receive, send)
