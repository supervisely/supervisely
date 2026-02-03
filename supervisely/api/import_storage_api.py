# coding: utf-8
"""Work with internal import storage in Supervisely."""

from supervisely.api.module_api import ModuleApiBase, ApiField


class ImportStorageApi(ModuleApiBase):
    def get_meta_by_hashes(self, hashes):
        """
        """
        response = self._api.post('import-storage.internal.meta.list', {ApiField.HASHES: hashes})
        return response.json()
