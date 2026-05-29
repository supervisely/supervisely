from types import SimpleNamespace

import pytest

import supervisely.app.fastapi.subapp as fastapi_subapp
from supervisely.app.singleton import Singleton


@pytest.fixture(autouse=True)
def reset_app_singletons(monkeypatch):
    monkeypatch.setenv("ENV", "development")
    Singleton._instances.clear()
    Singleton._nested_instances.clear()
    yield
    Singleton._instances.clear()
    Singleton._nested_instances.clear()


class RouterWithEventHandler:
    def __init__(self):
        self.events = []

    def add_event_handler(self, event_type, func):
        self.events.append((event_type, func))


class AppWithEventHandler:
    def __init__(self):
        self.router = SimpleNamespace()
        self.events = []

    def add_event_handler(self, event_type, func):
        self.events.append((event_type, func))


class RouterWithWebsocketRoute:
    def __init__(self):
        self.routes = []

    def add_websocket_route(self, path, route, name=None):
        self.routes.append((path, route, name))


class AppWithWebsocketRoute:
    def __init__(self):
        self.router = SimpleNamespace()
        self.routes = []

    def add_websocket_route(self, path, route, name=None):
        self.routes.append((path, route, name))


def test_add_event_handler_prefers_router_api():
    app = SimpleNamespace(router=RouterWithEventHandler())

    def startup():
        return None

    fastapi_subapp._add_event_handler(app, "startup", startup)

    assert app.router.events == [("startup", startup)]


def test_add_event_handler_falls_back_to_app_api():
    app = AppWithEventHandler()

    def shutdown():
        return None

    fastapi_subapp._add_event_handler(app, "shutdown", shutdown)

    assert app.events == [("shutdown", shutdown)]


def test_add_event_handler_falls_back_to_router_handler_lists():
    router = SimpleNamespace(on_startup=[], on_shutdown=[])
    app = SimpleNamespace(router=router)

    def startup():
        return None

    fastapi_subapp._add_event_handler(app, "startup", startup)

    assert router.on_startup == [startup]


def test_add_event_handler_raises_clear_error_without_supported_api():
    app = SimpleNamespace(router=SimpleNamespace())

    with pytest.raises(AttributeError, match="event registration API"):
        fastapi_subapp._add_event_handler(app, "startup", lambda: None)


def test_add_websocket_route_prefers_router_api():
    app = SimpleNamespace(router=RouterWithWebsocketRoute())
    endpoint = object()

    fastapi_subapp._add_websocket_route(app, "/hot-reload", endpoint, name="hot-reload")

    assert app.router.routes == [("/hot-reload", endpoint, "hot-reload")]


def test_add_websocket_route_falls_back_to_app_api():
    app = AppWithWebsocketRoute()
    endpoint = object()

    fastapi_subapp._add_websocket_route(app, "/hot-reload", endpoint, name="hot-reload")

    assert app.routes == [("/hot-reload", endpoint, "hot-reload")]


def test_add_websocket_route_raises_clear_error_without_supported_api():
    app = SimpleNamespace(router=SimpleNamespace())

    with pytest.raises(AttributeError, match="raw websocket route registration API"):
        fastapi_subapp._add_websocket_route(app, "/hot-reload", object(), name="hot-reload")


def test_enable_hot_reload_on_debug_uses_router_websocket_api(monkeypatch):
    class FakePath:
        def __init__(self, path):
            self.path = path

    class FakeHotReload:
        def __init__(self, paths):
            self.paths = paths
            self.startup = lambda: None
            self.shutdown = lambda: None

        async def __call__(self, scope, receive, send):
            return None

    fake_arel = SimpleNamespace(Path=FakePath, HotReload=FakeHotReload)
    app = SimpleNamespace(router=RouterWithWebsocketRoute())
    app.router.events = []
    app.router.add_event_handler = lambda event_type, func: app.router.events.append(
        (event_type, func)
    )

    monkeypatch.setattr(fastapi_subapp, "arel", fake_arel)
    monkeypatch.setattr(fastapi_subapp.sys, "gettrace", lambda: object())

    fastapi_subapp.enable_hot_reload_on_debug(app)

    assert len(app.router.routes) == 1
    assert app.router.routes[0][0] == "/hot-reload"
    assert app.router.routes[0][2] == "hot-reload"
    assert [event_type for event_type, _ in app.router.events] == ["startup", "shutdown"]


def test_reload_page_noops_when_hot_reload_disabled_without_arel(monkeypatch, tmp_path):
    monkeypatch.setattr(fastapi_subapp, "arel", None)

    app = fastapi_subapp.Application(
        templates_dir=str(tmp_path), hot_reload=True, __local__=True
    )

    assert app.hot_reload is None
    app.reload_page()
