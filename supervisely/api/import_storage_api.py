# coding: utf-8

from supervisely.api.module_api import ModuleApiBase, ApiField


class ImportStorageApi(ModuleApiBase):
    """Work with internal import storage in Supervisely."""

    def get_meta_by_hashes(self, hashes):
        """
        """
        response = self._api.post('import-storage.internal.meta.list', {ApiField.HASHES: hashes})
        return response.json()
