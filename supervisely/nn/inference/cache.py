import os
import shutil
from typing import Any, Dict, Optional, Type, Hashable
from cachetools import LRUCache, cached, Cache
from shelved_cache import PersistentCache
from threading import Lock

import supervisely as sly
from pathlib import Path


default_lru_params = {"maxsize": 128}


class PersistentImageLRUCache(LRUCache):
    __marker = object()

    def __init__(self, maxsize, filepath: Path, exist_ok: bool = False, getsizeof=None):
        super().__init__(maxsize)
        self._base_dir = filepath
        self._exist_ok = exist_ok

    def __getitem__(self, key: Any) -> Any:
        filepath = super(PersistentImageLRUCache, self).__getitem__(key)
        return sly.image.read(str(filepath))

    def __setitem__(self, key: Any, value: Any) -> None:
        if not self._base_dir.exists():
            self._base_dir.mkdir()

        filepath = self._base_dir / f"{str(key)}.png"
        super(PersistentImageLRUCache, self).__setitem__(key, filepath)

        if filepath.exists():
            sly.logger.debug(f"Rewrite image {str(filepath)}")
        sly.image.write(str(filepath), value)

    def pop(self, key, default=__marker):
        if key in self:
            filepath = super(PersistentImageLRUCache, self).__getitem__(key)
            value = self[key]
            del self[key]
            os.remove(filepath)
            sly.logger.debug(f"Remove {filepath} frame")
        elif default is self.__marker:
            raise KeyError(key)
        else:
            value = default
        return value

    def clear(self, rm_base_folder=True) -> None:
        while self.currsize > 0:
            self.popitem()
        if rm_base_folder:
            shutil.rmtree(self._base_dir)
