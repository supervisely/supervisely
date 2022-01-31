
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
    def __init__(self, field: Field, *args, **kwargs):
        super().__init__(*args, **kwargs)
        PatchableJson._field = field
        self._ws = WebsocketManager()
        self._last = dict(self)
        self._lock = asyncio.Lock()
        self._field = field

    def get_changes(self, patch=None):
        if patch is None:
            patch = self._get_patch()
        return {self._field: patch}

    def _get_patch(self):
        patch = jsonpatch.JsonPatch.from_diff(self._last, self)
        return patch

    async def _apply_patch(self, patch):
        async with self._lock:
            patch.apply(self._last, in_place=True)

    async def synchronize_changes(self):
        patch = self._get_patch()
        await self._apply_patch(patch)
        await self._ws.broadcast(self.get_changes(patch))

    @classmethod
    async def from_request(cls, request: Request):
        content = await request.json()
        d = content.get(cls._field, {})
        return cls(cls._field, d)


class LastStateJson(PatchableJson, metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        super().__init__(Field.STATE, *args, **kwargs)
    
    @classmethod
    async def from_request(cls, request: Request):
        content = await request.json()
        d = content.get(cls._field)
        last_state = cls()
        if d is not None:
            async with last_state._lock:
                last_state.clear()
                last_state.update(d)
        return last_state
    
    @classmethod
    async def replace(cls, request: Request): 
        # update method already exists in dict
        return await cls.from_request(request)


class ContextJson(PatchableJson):
    def __init__(self, *args, **kwargs):
        super().__init__(Field.CONTEXT, *args, **kwargs)


class StateJson(PatchableJson):
    def __init__(self, *args, **kwargs):
        super().__init__(Field.STATE, *args, **kwargs)
    
    def _apply_patch(self, patch):
        super()._apply_patch(patch)
        LastStateJson()._apply_patch(patch)


class DataJson(PatchableJson, metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        super().__init__(Field.DATA, *args, **kwargs)
    
    @classmethod
    async def from_request(cls, request: Request):
        raise RuntimeError(f"""Request from Supervisely App never contains \"{cls._field}\" field. Every request from app contains by default current state and context""")


