import os
import signal
import psutil
import asyncio
from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi import Response


class ShutdownMiddleware:
    def __init__(
        self, app: ASGIApp, path: str = "/shutdown"
    ) -> None:
        self.app = app
        self.path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            if scope["path"] == self.path:
                scope["app"].state.STOPPED = True
                self.shutdown()
            if hasattr(scope["app"].state, 'STOPPED'):
                stopped = scope["app"].state.STOPPED
                if stopped:
                    # PlainTextResponse("Invalid host header", status_code=400)
                    response = Response(content="Server is being shut down", status_code=403)
                    return await response(scope, receive, send) 
        
        await self.app(scope, receive, send)

    def shutdown(self):
        # https://github.com/tiangolo/fastapi/issues/1509
        current_process = psutil.Process(os.getpid())
        current_process.send_signal(signal.SIGINT) # emit ctrl + c
