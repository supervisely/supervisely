from starlette.types import ASGIApp, Receive, Scope, Send
from supervisely.fastapi_helpers import WebsocketManager 


# class WebsocketMiddleware:
#     def __init__(
#         self, 
#         app: ASGIApp,
#         path: str = '/sly-app-ws',
#     ) -> None:
#         self.app = app
#         self.path = path
#         fast_api_app = app.app.dependency_overrides_provider
#         WebsocketManager().set_app(fast_api_app)

#     async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
#         await self.app(scope, receive, send)
