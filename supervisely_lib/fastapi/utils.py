import os
import signal
import psutil
import asyncio
from fastapi import FastAPI, HTTPException, Request


async def graceful_shutdown(app: FastAPI):
    app.state.STOPPED = True
    await asyncio.sleep(10)

    # https://github.com/tiangolo/fastapi/issues/1509
    current_process = psutil.Process(os.getpid())
    current_process.send_signal(signal.SIGINT) # emit ctrl + c


# https://github.com/tiangolo/fastapi/issues/1501
# https://github.com/tiangolo/fastapi/issues/1501#issuecomment-638219871


# class ShutdownMiddleware(BaseHTTPMiddleware):
#     def __init__(self, param=1, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.param = param
#         self.stopped = False

#     async def dispatch(self, request: Request, call_next):
#         print("!!!-> ", self.param)
#         if self.stopped:
#             raise HTTPException(status_code=403, detail="Server is being shut down")
#         response = await call_next(request)    
#         return response


# class ShutdownMiddlewareCallable:
#     def __init__(self, app, param=1):
#         self.app = app
#         self.param = param

#     async def __call__(self, request: Request, call_next):
#         print("!!!-> ", self.param)
#         # if hasattr(request.app.state, 'STOPPED'):
#         #     print("!!!-> state ", request.app.state.STOPPED)
#         #     if request.app.state.STOPPED:
#         #         raise HTTPException(status_code=403, detail="Server is being shut down")
#         # response = await call_next(request)    
#         # return response

# # https://github.com/tiangolo/fastapi/issues/1646
# class RoomEventMiddleware:  # pylint: disable=too-few-public-methods
#     def __init__(self, app):
#         self._app = app
#         self._room = Room()

#     async def __call__(self, scope: Scope, receive: Receive, send: Send):
#         if scope["type"] in ("lifespan", "http", "websocket"):
#             scope["room"] = self._room
#         await self._app(scope, receive, send)

# app.add_middleware(RoomEventMiddleware)



class ShutdownMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, param='Example'):
        super().__init__(app)
        self.param = param

    async def dispatch(request: Request, call_next):
        print("!!!-> ", self.param)
        if hasattr(request.app.state, 'STOPPED'):
            print("!!!-> state ", request.app.state.STOPPED)
            if request.app.state.STOPPED:
                raise HTTPException(status_code=403, detail="Server is being shut down")
        response = await call_next(request)    
        return response