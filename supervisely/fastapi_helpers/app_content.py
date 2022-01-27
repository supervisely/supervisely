
import enum
import jsonpatch
import asyncio
from fastapi import FastAPI
from fastapi import Request
from supervisely.fastapi_helpers.singleton import Singleton
from supervisely.fastapi_helpers.websocket import WebsocketManager


class Field(str, enum.Enum):
    STATE = 'state'
    DATA = 'data'


class PatchableJson(dict):
    _app = None

    def __init__(self, app: FastAPI, field: Field, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._app is None:
            self._app = app
        self._ws = WebsocketManager(app)
        self._field = field
        self._last = dict(self)
        self._lock = asyncio.Lock()

    def _get_patch(self):
        patch = jsonpatch.JsonPatch.from_diff(self._last, self)
        return patch

    async def _apply_patch(self, patch):
        async with self._lock:
            patch.apply(self._last, in_place=True)

    async def synchronize_changes(self):
        patch = self._get_patch()
        await self._apply_patch(patch)
        await self._ws.broadcast({self._field: patch})


class LastStateJson(PatchableJson, metaclass=Singleton):
    def __init__(self, app: FastAPI, *args, **kwargs):
        super().__init__(app, Field.STATE, *args, **kwargs)


class StateJson(PatchableJson):
    def __init__(self, app: FastAPI, *args, **kwargs):
        super().__init__(app, Field.STATE, *args, **kwargs)
    
    def _apply_patch(self, patch):
        super()._apply_patch(patch)
        LastStateJson()._apply_patch(patch)

    @classmethod
    async def from_request(cls, request: Request):
        content = await request.json()
        state = content.get(Field.STATE, {})
        return cls(cls._app, state)


class DataJson(PatchableJson, metaclass=Singleton):
    def __init__(self, app: FastAPI, *args, **kwargs):
        super().__init__(app, Field.DATA, *args, **kwargs)



