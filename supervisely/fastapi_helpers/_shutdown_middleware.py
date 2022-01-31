import os
import signal
import psutil
from fastapi import FastAPI
from starlette.types import ASGIApp
from  fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint, DispatchFunction


# class ShutdownMiddleware(BaseHTTPMiddleware):
#     def __init__(
#         self, 
#         app: ASGIApp, 
#         dispatch: DispatchFunction = None, 
#         path: str = '/sly-app-shutdown'
#     ) -> None:
#         super().__init__(app, dispatch)
#         self.path = path

#     async def dispatch(
#         self, request: Request, call_next: RequestResponseEndpoint
#     ) -> Response:
#         if request.url.path == self.path and not hasattr(request.app.state, 'STOPPED'):
#             response = JSONResponse(content="Server will be shutdown")
#             await shutdown(request.app)
#             return response
#         elif hasattr(request.app.state, 'STOPPED') and request.app.state.STOPPED:
#             response = JSONResponse(content="Server is being shut down", status_code=403)
#             return response 
#         response = await call_next(request)
#         return response


async def shutdown(app: FastAPI):
    # setattr(app.state, "STOPPED", True)
    
    # for debug
    # import asyncio
    # await asyncio.sleep(10)
    current_process = psutil.Process(os.getpid())
    current_process.send_signal(signal.SIGINT) # emit ctrl + c
