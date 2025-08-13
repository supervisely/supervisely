from typing import List, Literal

from supervisely.api.api import Api
from supervisely.api.module_api import ApiField, ModuleApi


class ModelApiField:
    NAME = "name"
    FRAMEWORK = "framework"
    TASK_TYPE = "task"
    MODALITY = "modality"
    TRAIN_MODULE_ID = "trainModuleId"
    SERVE_MODULE_ID = "serveModuleId"
    ARCHITECTURE = "architecture"
    PRETRAINED = "pretrained"
    NUM_CLASSES = "numClasses"
    SIZE = "size"
    PARAMS_M = "paramsM"
    GFLOPS = "GFLOPs"
    TAGS = "tags"
    RUNTIMES = "runtimes"
    FILES = "files"
    SPEED_TESTS = "speedTests"
    EVALUATION = "evaluation"


class EcosystemModelsApi(ModuleApi):

    def __init__(self, api: Api):
        self._api = api

    def _convert_json_info(self, json_info):
        return json_info

    def get_list_all_pages(
        self,
        method,
        data,
        progress_cb=None,
        convert_json_info_cb=None,
        limit: int = None,
        return_first_response: bool = False,
    ):
        """
        Get list of all or limited quantity entities from the Supervisely server.

        :param method: Request method name
        :type method: str
        :param data: Dictionary with request body info
        :type data: dict
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Progress, optional
        :param convert_json_info_cb: Function for convert json info
        :type convert_json_info_cb: Callable, optional
        :param limit: Number of entity to retrieve
        :type limit: int, optional
        :param return_first_response: Specify if return first response
        :type return_first_response: bool, optional
        """

        if convert_json_info_cb is None:
            convert_func = self._convert_json_info
        else:
            convert_func = convert_json_info_cb

        if ApiField.SORT not in data:
            data = self._add_sort_param(data)
        first_response = self._api.get(method, {}, data=data).json()
        total = first_response["total"]
        per_page = first_response["perPage"]
        pages_count = first_response["pagesCount"]

        limit_exceeded = False
        results = first_response["entities"]
        if limit is not None and len(results) > limit:
            limit_exceeded = True

        if progress_cb is not None:
            progress_cb(len(results))
        if (pages_count == 1 and len(results) == total) or limit_exceeded is True:
            pass
        else:
            for page_idx in range(2, pages_count + 1):
                temp_resp = self._api.get(
                    method, {}, data={**data, "page": page_idx, "per_page": per_page}
                )
                temp_items = temp_resp.json()["entities"]
                results.extend(temp_items)
                if progress_cb is not None:
                    progress_cb(len(temp_items))
                if limit is not None and len(results) > limit:
                    limit_exceeded = True
                    break

            if len(results) != total and limit is None:
                raise RuntimeError(
                    "Method {!r}: error during pagination, some items are missed".format(method)
                )

        if limit is not None:
            results = results[:limit]
        if return_first_response:
            return [convert_func(item) for item in results], first_response
        return [convert_func(item) for item in results]

    def list_models(self, local=False):
        method = "ecosystem.models.list"
        data = {"localModels": local}
        return self.get_list_all_pages(method, data=data)

    def add(
        self,
        name: str,
        framework: str,
        task_type: str,
        tain_module_id: int,
        serve_module_id: int,
        modality: Literal["images", "videos"] = "images",
        architecture: str = None,
        pretrained: bool = False,
        num_classes: int = None,
        size: int = None,
        params_m: int = None,
        gflops: float = None,
        tags: List[str] = None,
        runtimes: List[str] = None,
        files: List[str] = None,
        speed_tests: List = None,
        evaluation: dict = None,
    ):
        method = "ecosystem.models.add"
        data = {
            ModelApiField.NAME: name,
            ModelApiField.FRAMEWORK: framework,
            ModelApiField.TASK_TYPE: task_type,
            ModelApiField.MODALITY: modality,
            ModelApiField.TRAIN_MODULE_ID: tain_module_id,
            ModelApiField.SERVE_MODULE_ID: serve_module_id,
        }
        optional_fields = {
            ModelApiField.ARCHITECTURE: architecture,
            ModelApiField.PRETRAINED: pretrained,
            ModelApiField.NUM_CLASSES: num_classes,
            ModelApiField.SIZE: size,
            ModelApiField.PARAMS_M: params_m,
            ModelApiField.GFLOPS: gflops,
            ModelApiField.TAGS: tags,
            ModelApiField.RUNTIMES: runtimes,
            ModelApiField.FILES: files,
            ModelApiField.SPEED_TESTS: speed_tests,
            ModelApiField.EVALUATION: evaluation,
        }
        for key, value in optional_fields.items():
            if value is not None:
                data[key] = value
        return self._api.post(method, data=data)

    def update_model(
        self,
        model_id: int,
        name: str = None,
        framework: str = None,
        task_type: str = None,
        tain_module_id: int = None,
        serve_module_id: int = None,
        modality: Literal["images", "videos"] = None,
        architecture: str = None,
        pretrained: bool = False,
        num_classes: int = None,
        size: int = None,
        params_m: int = None,
        gflops: float = None,
        tags: List[str] = None,
        runtimes: List[str] = None,
        files: List[str] = None,
        speed_tests: List = None,
        evaluation: dict = None,
    ):
        data = {
            ModelApiField.NAME: name,
            ModelApiField.FRAMEWORK: framework,
            ModelApiField.TASK_TYPE: task_type,
            ModelApiField.MODALITY: modality,
            ModelApiField.TRAIN_MODULE_ID: tain_module_id,
            ModelApiField.SERVE_MODULE_ID: serve_module_id,
            ModelApiField.ARCHITECTURE: architecture,
            ModelApiField.PRETRAINED: pretrained,
            ModelApiField.NUM_CLASSES: num_classes,
            ModelApiField.SIZE: size,
            ModelApiField.PARAMS_M: params_m,
            ModelApiField.GFLOPS: gflops,
            ModelApiField.TAGS: tags,
            ModelApiField.RUNTIMES: runtimes,
            ModelApiField.FILES: files,
            ModelApiField.SPEED_TESTS: speed_tests,
            ModelApiField.EVALUATION: evaluation,
        }
        data = {k: v for k, v in data.items() if v is not None}
        method = "ecosystem.models.update"
        data["id"] = model_id
        return self._api.post(method, data=data)
