from urllib.request import Request
from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi import Request
from fastapi.responses import JSONResponse


class StateMiddleware:
    def __init__(
        self, 
        app: ASGIApp,
        state: dict,
        path = "/sly-app-state"
    ) -> None:
        self.app = app
        self.state = state
        self.path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http": 
            if scope["path"] == self.path:
                response = JSONResponse(content=self.state)
                return await response(scope, receive, send) 
            else:
                request = Request(scope, receive, send)
                current_state = await request.json()
                self.state.clear()
                self.state.update(current_state)
        await self.app(scope, receive, send)
