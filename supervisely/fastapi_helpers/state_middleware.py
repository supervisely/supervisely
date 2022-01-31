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
            response = JSONResponse(content=dict(last_state))
            return response
        
        last_state.replace(request)
        response = await call_next(request)
        return response
