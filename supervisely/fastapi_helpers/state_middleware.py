from urllib.request import Request
from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi import Request
from fastapi.responses import JSONResponse


class StateMiddleware:
    def __init__(
        self, 
        app: ASGIApp,
        path = "/sly-app-state"
    ) -> None:
        self.app = app
        self.path = path
        

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] == self.path:
            response = JSONResponse(content=self.state)
            return await response(scope, receive, send) 
        else:
            request = Request(scope, receive, send)
            #TODO: if json body
            request_json = await request.json()
            if "state" in request_json:
                self.state.clear()
                self.state.update(request_json["state"])
            return await self.app(scope, receive, send)
