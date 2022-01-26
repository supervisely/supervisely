import os
import types
import signal
import psutil
from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi import Response


class ShutdownMiddleware:
    def __init__(
        self, 
        app: ASGIApp,
        path: str = '/sly-app-shutdown',
    ) -> None:
        self.app = app
        self.path = path
        self._server = app.app.dependency_overrides_provider
        self._server.shutdown = types.MethodType( graceful_shutdown, self._server )
        # https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
        # setattr(FastAPI, 'shutdown', graceful_shutdown)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] == self.path and not hasattr(scope["app"].state, 'STOPPED'):
            scope["app"].state.STOPPED = True
            await Response()(scope, receive, send) 
            return self.app.graceful_shutdown()
        elif hasattr(scope["app"].state, 'STOPPED') and scope["app"].state.STOPPED:
            response = Response(content="Server is being shut down", status_code=403)
            return await response(scope, receive, send) 
        await self.app(scope, receive, send)


def graceful_shutdown(self):
    # for debug
    # import asyncio
    # await asyncio.sleep(10)
    current_process = psutil.Process(os.getpid())
    current_process.send_signal(signal.SIGINT) # emit ctrl + c