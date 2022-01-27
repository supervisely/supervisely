
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
    CONTEXT = 'context'


class PatchableJson(dict):
    _app = None
    _field = None
    
    def __init__(self, app: FastAPI, field: Field, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if PatchableJson._app is None:
            PatchableJson._app = app
        PatchableJson._field = field
        self._ws = WebsocketManager(app)
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

    @classmethod
    async def from_request(cls, request: Request):
        content = await request.json()
        d = content.get(cls._field, {})
        return cls(cls._app, d)


class LastStateJson(PatchableJson, metaclass=Singleton):
    def __init__(self, app: FastAPI, *args, **kwargs):
        super().__init__(app, Field.STATE, *args, **kwargs)
    
    @classmethod
    async def from_request(cls, request: Request):
        content = await request.json()
        d = content.get(cls._field, {})
        last_state = cls(cls._app)
        async with last_state._lock:
            last_state.clear()
            last_state.update(d)
        return last_state


class ContextJson(PatchableJson):
    def __init__(self, app: FastAPI, *args, **kwargs):
        super().__init__(app, Field.CONTEXT, *args, **kwargs)


class StateJson(PatchableJson):
    def __init__(self, app: FastAPI, *args, **kwargs):
        super().__init__(app, Field.STATE, *args, **kwargs)
    
    def _apply_patch(self, patch):
        super()._apply_patch(patch)
        LastStateJson()._apply_patch(patch)


class DataJson(PatchableJson, metaclass=Singleton):
    def __init__(self, app: FastAPI, *args, **kwargs):
        super().__init__(app, Field.DATA, *args, **kwargs)
    
    @classmethod
    async def from_request(cls, request: Request):
        raise RuntimeError(f"""Request from Supervisely App never contains \"{cls._field}\" field. Every request from app contains by default current state and context""")


