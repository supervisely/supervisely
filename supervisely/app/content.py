from __future__ import annotations
import copy
import os
import enum
import json
import jsonpatch
import asyncio
from fastapi import Request
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.io.fs import dir_exists, mkdir
from supervisely.sly_logger import logger
from supervisely.app.singleton import Singleton
from supervisely.app.fastapi import run_sync


class Field(str, enum.Enum):
    STATE = "state"
    DATA = "data"
    CONTEXT = "context"


def get_data_dir():
    dir = None

    task_id = os.environ.get("TASK_ID")
    if task_id is not None:
        dir = f"/sessions/{task_id}"

    keys = ["SLY_APP_DATA_DIR", "DEBUG_APP_DIR"]
    for key in keys:
        value = os.environ.get(key)
        if value is not None:
            dir = value
            logger.debug(f"Load dir from evn {key}={value}")
            break
    if dir is None:
        raise ValueError(f"One of the env variables have to be defined: {[*keys, 'TASK_ID']}")

    if dir_exists(dir) is False:
        logger.info(f"App data directory {dir} doesn't exist. Will be made automatically.")
        mkdir(dir)
    return dir


def get_synced_data_dir():
    dir = "/sly-app-data"

    keys = ["SLY_APP_DATA_DIR", "DEBUG_APP_DIR"]
    for key in keys:
        value = os.environ.get(key)
        if value is not None:
            dir = value
            logger.debug(f"Load dir from evn {key}={value}")
            break

    if dir_exists(dir) is False:
        logger.info(f"Synced app data directory {dir} doesn't exist. Will be made automatically.")
        mkdir(dir)
    return dir


class _PatchableJson(dict):
    def __init__(self, field: Field, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ws = WebsocketManager()
        self._last = copy.deepcopy(dict(self))
        self._lock = asyncio.Lock()
        self._field = field.value

    def get_changes(self, patch=None):
        if patch is None:
            patch = self._get_patch()
        return {self._field: json.loads(patch.to_string())}

    def _get_patch(self):
        patch = jsonpatch.JsonPatch.from_diff(self._last, self)
        return patch

    async def _apply_patch(self, patch):
        async with self._lock:
            patch.apply(self._last, in_place=True)
            self._last = copy.deepcopy(self._last)

    async def synchronize_changes(self):
        patch = self._get_patch()
        await self._apply_patch(patch)
        await self._ws.broadcast(self.get_changes(patch))

    async def send_changes_async(self):
        await self.synchronize_changes()

    def send_changes(self):
        run_sync(self.synchronize_changes())

    def raise_for_key(self, key: str):
        if key in self:
            raise KeyError(f"Key {key} already exists in {self._field}")


class StateJson(_PatchableJson, metaclass=Singleton):
    _global_lock: asyncio.Lock = None

    def __init__(self, *args, **kwargs):
        if StateJson._global_lock is None:
            StateJson._global_lock = asyncio.Lock()
        super().__init__(Field.STATE, *args, **kwargs)

    async def _apply_patch(self, patch):
        await super()._apply_patch(patch)
        # @TODO: _replace_global to patching for optimization
        await StateJson._replace_global(dict(self))

    @classmethod
    async def from_request(cls, request: Request) -> StateJson:
        if "application/json" not in request.headers.get("Content-Type", ""):
            return None
        content = await request.json()
        d = content.get(Field.STATE, {})
        await cls._replace_global(d)
        return cls(d, __local__=True)

    @classmethod
    async def _replace_global(cls, d: dict):
        async with cls._global_lock:
            global_state = cls()
            global_state.clear()
            global_state.update(copy.deepcopy(d))
            global_state._last = copy.deepcopy(d)


class DataJson(_PatchableJson, metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        super().__init__(Field.DATA, *args, **kwargs)
