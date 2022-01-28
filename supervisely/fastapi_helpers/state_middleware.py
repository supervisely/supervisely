from starlette.types import ASGIApp
from  fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint, DispatchFunction
from supervisely.fastapi_helpers.app_content import LastStateJson


class StateMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app: ASGIApp, 
        dispatch: DispatchFunction = None, 
        path: str = '/sly-app-state'
    ) -> None:
        super().__init__(app, dispatch)
        self.path = path

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        last_state = LastStateJson()

        if request.url.path == self.path:
            response = JSONResponse(content={last_state._field: dict(last_state)})
            return response
        
        last_state.replace(request)
        response = await call_next(request)
        return response


# low level incorrect implementation
# class StateMiddleware:
#     def __init__(
#         self, 
#         app: ASGIApp,
#         path = "/sly-app-state"
#     ) -> None:
#         self.app = app
#         self.path = path
#         LastStateJson()
        
#     async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
#         if scope["type"] != "http" or scope["method"] == "GET":
#             return await self.app(scope, receive, send)
        
#         #@TODO: error implementation here
#         # request = Request(scope, receive, send)
#         # await LastStateJson.replace(request)

#         # request = Request(scope, receive, send)
#         # await LastStateJson.replace(request)

#         if scope["path"] == self.path:
#             last_state = LastStateJson()
#             response = JSONResponse(content={last_state._field: dict(last_state)})
#             return await response(scope, receive, send) 
        
#         return await self.app(scope, receive, send)
