
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


class _PatchableJson(dict):
    def __init__(self, field: Field, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        return cls(d)


class LastStateJson(_PatchableJson, metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        super().__init__(Field.STATE, *args, **kwargs)
    
    @classmethod
    async def from_request(cls, request: Request):
        content = {}
        # try:
        #     # for '/' endpoint
            # content = await request.json()
        # except ValueError:
        #     return cls()

        # content = await request.json()

        last_state = cls()
        d = content.get(last_state._field)
        if d is not None:
            async with last_state._lock:
                last_state.clear()
                last_state.update(d)
        return last_state
    
    @classmethod
    async def replace(cls, request: Request): 
        # update method already exists in dict
        return await cls.from_request(request)


class ContextJson(_PatchableJson):
    def __init__(self, *args, **kwargs):
        super().__init__(Field.CONTEXT, *args, **kwargs)


class StateJson(_PatchableJson):
    def __init__(self, *args, **kwargs):
        super().__init__(Field.STATE, *args, **kwargs)
    
    async def _apply_patch(self, patch):
        await super()._apply_patch(patch)
        await LastStateJson()._apply_patch(patch)

    @classmethod
    async def from_request(cls, request: Request):
        state_json = await super().from_request(request)
        LastStateJson.replace(request)
        return state_json


class DataJson(_PatchableJson, metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        super().__init__(Field.DATA, *args, **kwargs)
    
    @classmethod
    async def from_request(cls, request: Request):
        raise RuntimeError(f"""Request from Supervisely App never contains \"{cls._field}\" field. Every request from app contains by default current state and context""")


