import os
import signal
import psutil
import time
from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi import Response, Request


class ShutdownMiddleware:
    def __init__(
        self, app: ASGIApp
    ) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            if hasattr(scope["app"].state, 'STOPPED') and scope["app"].state.STOPPED:
                response = Response(content="Server is being shut down", status_code=403)
                return await response(scope, receive, send) 
        await self.app(scope, receive, send)


def shutdown_fastapi(request: Request):
    request.app.state.STOPPED = True
    # time.sleep(10)
    current_process = psutil.Process(os.getpid())
    current_process.send_signal(signal.SIGINT) # emit ctrl + c