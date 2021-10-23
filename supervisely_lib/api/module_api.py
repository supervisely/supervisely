# coding: utf-8

import requests
from collections import namedtuple
from copy import deepcopy
from supervisely_lib._utils import batched

from supervisely_lib._utils import camel_to_snake


class ApiField:
    ID =                'id'
    NAME =              'name'
    DESCRIPTION =       'description'
    CREATED_AT =        'createdAt'
    UPDATED_AT =        'updatedAt'
    ROLE =              'role'
    TEAM_ID =           'teamId'
    WORKSPACE_ID =      'workspaceId'
    CONFIG =            'config'
    SIZE =              'size'
    PLUGIN_ID =         'pluginId'
    PLUGIN_VERSION =    'pluginVersion'
    HASH =              'hash'
    STATUS =            'status'
    ONLY_TRAIN =        'onlyTrain'
    USER_ID =           'userId'
    README =            'readme'
    PROJECT_ID =        'projectId'
    IMAGES_COUNT =      'imagesCount'
    ITEMS_COUNT =       'itemsCount'
    DATASET_ID =        'datasetId'
    LINK =              'link'
    WIDTH =             'width'
    HEIGHT =            'height'
    WEIGHTS_LOCATION =  'weightsLocation'
    IMAGE_ID =          'imageId'
    IMAGE_NAME =        'imageName'
    ANNOTATION =        'annotation'
    TASK_ID =           'taskId'
    LABELS_COUNT =      'labelsCount'
    FILTER =            'filter'
    META =              'meta'
    FILE_META =         'fileMeta'
    SHARED_LINK =       'sharedLinkToken'
    EXPLORE_PATH =      'explorePath'
    MIME =              'mime'
    EXT =               'ext'
    TYPE =              'type'
    DEFAULT_VERSION =   'defaultVersion'
    DOCKER_IMAGE =      'dockerImage'
    CONFIGS =           'configs'
    VERSIONS =          'versions'
    VERSION =           'version'
    TOKEN =             'token'
    CAPABILITIES =      'capabilities'
    STARTED_AT =        'startedAt'
    FINISHED_AT =       'finishedAt'
    AGENT_ID =          'agentId'
    MODEL_ID =          'modelId'
    RESTART_POLICY =    'restartPolicy'
    SETTINGS =          'settings'
    SORT =              'sort'
    SORT_ORDER =        'sort_order'
    IMAGES =            'images'
    IMAGE_IDS =         'imageIds'
    ANNOTATIONS =       'annotations'
    EMAIL =             'email'
    LOGIN =             'login'
    LOGINS =            'logins'
    DISABLED =          'disabled'
    LAST_LOGIN =        'lastLogin'
    PASSWORD =          'password'
    ROLE_ID =           'roleId'
    IS_RESTRICTED =     'isRestricted'
    DISABLE =           'disable'
    TEAMS =             'teams'
    USER_IDS =          'userIds'
    PROJECT_NAME =      (['projectTitle'], 'project_name')
    DATASET_NAME =      (['datasetTitle'], 'dataset_name')
    WORKSPACE_NAME =    (['workspaceTitle'], 'workspace_name')
    CREATED_BY_ID =     (['createdBy'], 'created_by_id')
    CREATED_BY_LOGIN =  (['managerLogin'], 'created_by_login')
    ASSIGNED_TO_ID =    (['userId'], 'assigned_to_id')
    ASSIGNED_TO_LOGIN = (['labelerLogin'], 'assigned_to_login')
    FINISHED_IMAGES_COUNT = 'finishedImagesCount'
    PROGRESS_IMAGES_COUNT = 'progressImagesCount'
    CLASSES_TO_LABEL =  (['meta', 'classes'], 'classes_to_label')
    TAGS_TO_LABEL =     (['meta', 'projectTags'], 'tags_to_label')
    IMAGES_RANGE =      (['meta', 'range'], 'images_range')
    REJECTED_IMAGES_COUNT = 'rejectedImagesCount'
    ACCEPTED_IMAGES_COUNT = 'acceptedImagesCount'
    OBJECTS_LIMIT_PER_IMAGE = (['meta', 'imageFiguresLimit'], 'objects_limit_per_image')
    TAGS_LIMIT_PER_IMAGE = (['meta', 'imageTagsLimit'], 'tags_limit_per_image')
    FILTER_IMAGES_BY_TAGS = (['meta', 'imageTags'], 'filter_images_by_tags')
    INCLUDE_IMAGES_WITH_TAGS = ([], 'include_images_with_tags')
    EXCLUDE_IMAGES_WITH_TAGS = ([], 'exclude_images_with_tags')
    SIZEB =             (['size'], 'sizeb')
    FRAMES_COUNT =      (['fileMeta', 'framesCount'], 'frames_count')
    PATH_ORIGINAL =     'pathOriginal'
    OBJECTS_COUNT =     'objectsCount'
    FRAMES_TO_TIMECODES = (['fileMeta', 'framesToTimecodes'], 'frames_to_timecodes')
    TAGS =              'tags'
    VIDEO_ID =          'videoId'
    FRAME_INDEX =       (['meta', 'frame'], 'frame_index')
    LABELING_TOOL =     'tool'
    GEOMETRY_TYPE =     'geometryType'
    GEOMETRY =          'geometry'
    OBJECT_ID =         'objectId'
    FRAME =             'frame'
    #FIGURES_COUNT =     (['labelsCount'], 'figures_count')
    STREAMS =           'streams'
    VIDEO_IDS =         'videoIds'
    FRAME_WIDTH =       (['fileMeta', 'width'], 'frame_width')
    FRAME_HEIGHT =      (['fileMeta', 'height'], 'frame_height')
    VIDEO_NAME =        'videoName'
    FRAME_RANGE =       'frameRange'
    TRACK_ID =          'trackId'
    PROGRESS =          'progress'
    CURRENT =           'current'
    TOTAL =             'total'
    STOPPED =           'stopped'
    VIDEOS =            'videos'
    FILENAME =          'filename'
    SHAPE =             'shape'
    COLOR =             'color'
    CLASS_ID =          'classId'
    ENTITY_ID =         'entityId'
    ANNOTATION_OBJECTS = 'annotationObjects'
    TAG_ID =            'tagId'
    ERROR =             'error'
    MESSAGE =           'message'
    CONTENT =           'content'
    FIGURES =           'figures'
    LAYOUT =            'layout'
    WIDGETS =           'widgets'
    CLOUD_MIME =        (['fileMeta', 'mime'], 'cloud_mime')
    PREVIEW =           'preview'
    FIGURES_COUNT =     'figuresCount'
    ANN_OBJECTS_COUNT =     (['annotationObjectsCount'], 'objects_count')
    POINTCLOUD_ID =     'pointCloudId'
    POINTCLOUD_IDS =    'pointCloudIds'
    POINTCLOUDS =       'pointClouds'
    ADVANCED =          'advanced'
    IGNORE_AGENT =      'ignoreAgent'
    SCRIPT =            'script'
    LOGS =              'logs'
    FILES =             'files'
    HASHES =            'hashes'
    SUBTITLE =          'subtitle'
    COMMAND =           'command'
    DEFAULT_VALUE =     'defaultValue'
    TITLE =             'title'
    AREA =              'area'
    OPTIONS =           'options'
    REPORT_ID =         'reportId'
    WIDGET =            'widget'
    LAYOUT =            'layout'
    PAYLOAD =           'payload'
    FIELD =             'field'
    FIELDS =            'fields'
    APPEND =            'append'
    WITH_CUSTOM_DATA =  'withCustomBigData'
    PATH =              'path'
    SESSION_ID =        'sessionId'
    ACTION =            'action'
    FIGURE_ID =         'figureId'
    VALUE =             'value'
    ZOOM_FACTOR =       'zoomFactor'
    FULL_STORAGE_URL =  'fullStorageUrl'
    REVIEWER_ID =       'reviewerId'
    REVIEWER_LOGIN =    'reviewerLogin'
    RECURSIVE =         'recursive'
    ECOSYSTEM_ITEM_ID = 'moduleId'
    APP_ID =            'appId'
    PROJECT =           'project'
    OUTPUT =            'output'
    REFERENCE_IMAGE_URL='referenceImageUrl'
    GENERAL =           'general'
    ENTITIES =          'entities'
    STORAGE_PATH =      'storagePath'
    EXT2 =              (['meta', 'ext'], 'ext')
    MIME2 =             (['meta', 'mime'], 'mime')
    SIZEB2 =            (['meta', 'size'], 'sizeb')
    JOB_ID =            'jobId'
    DATASETS_COUNT =    'datasetsCount'
    CUSTOM_DATA =       'customData'
    CONTEXT =           'context'
    STATE =             'state'
    IDS =               'ids'
    DATE =              'date'


def _get_single_item(items):
    if len(items) == 0:
        return None
    if len(items) > 1:
        raise RuntimeError('There are several items with the same name')
    return items[0]


class _JsonConvertibleModule:
    def _convert_json_info(self, info: dict, skip_missing=False):
        raise NotImplementedError()


class ModuleApiBase(_JsonConvertibleModule):
    MAX_WAIT_ATTEMPTS = 999
    WAIT_ATTEMPT_TIMEOUT_SEC = 1

    @staticmethod
    def info_sequence():
        raise NotImplementedError()

    @staticmethod
    def info_tuple_name():
        raise NotImplementedError()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        try:
            field_names = []
            for name in cls.info_sequence():
                if type(name) is str:
                    field_names.append(camel_to_snake(name))
                elif type(name) is tuple and type(name[1]) is str:
                    field_names.append(name[1])
                else:
                    raise RuntimeError('Can not parse field {!r}'.format(name))
            cls.InfoType = namedtuple(cls.info_tuple_name(), field_names)
        except NotImplementedError:
            pass

    def __init__(self, api):
        self._api = api

    def _add_sort_param(self, data):
        results = deepcopy(data)
        results[ApiField.SORT] = ApiField.ID
        results[ApiField.SORT_ORDER] = 'asc'  # @TODO: move to enum
        return results

    def get_list_all_pages(self, method, data, progress_cb=None, convert_json_info_cb=None):
        if convert_json_info_cb is None:
            convert_func = self._convert_json_info
        else:
            convert_func = convert_json_info_cb

        if ApiField.SORT not in data:
            data = self._add_sort_param(data)
        first_response = self._api.post(method, data).json()
        total = first_response['total']
        per_page = first_response['perPage']
        pages_count = first_response['pagesCount']

        results = first_response['entities']
        if progress_cb is not None:
            progress_cb(len(results))
        if pages_count == 1 and len(first_response['entities']) == total:
            pass
        else:
            for page_idx in range(2, pages_count + 1):
                temp_resp = self._api.post(method, {**data, 'page': page_idx, 'per_page': per_page})
                temp_items = temp_resp.json()['entities']
                results.extend(temp_items)
                if progress_cb is not None:
                    progress_cb(len(temp_items))
            if len(results) != total:
                raise RuntimeError('Method {!r}: error during pagination, some items are missed'.format(method))

        return [convert_func(item) for item in results]

    @staticmethod
    def _get_info_by_name(get_info_by_filters_fn, name):
        filters = [{"field": ApiField.NAME, "operator": "=", "value": name}]
        return get_info_by_filters_fn(filters)

    def get_info_by_id(self, id):
        raise NotImplementedError()

    @staticmethod
    def _get_free_name(exist_check_fn, name):
        res_title = name
        suffix = 1
        while exist_check_fn(res_title):
            res_title = '{}_{:03d}'.format(name, suffix)
            suffix += 1
        return res_title

    def _convert_json_info(self, info: dict, skip_missing=False):
        def _get_value(dict, field_name, skip_missing):
            if skip_missing is True:
                return dict.get(field_name, None)
            else:
                return dict[field_name]

        if info is None:
            return None
        else:
            field_values = []
            for field_name in self.info_sequence():
                if type(field_name) is str:
                    field_values.append(_get_value(info, field_name, skip_missing))
                elif type(field_name) is tuple:
                    value = None
                    for sub_name in field_name[0]:
                        if value is None:
                            value = _get_value(info, sub_name, skip_missing)
                        else:
                            value = _get_value(value, sub_name, skip_missing)
                    field_values.append(value)
                else:
                    raise RuntimeError('Can not parse field {!r}'.format(field_name))
            return self.InfoType(*field_values)

    def _get_response_by_id(self, id, method, id_field):
        try:
            return self._api.post(method, {id_field: id})
        except requests.exceptions.HTTPError as error:
            if error.response.status_code == 404:
                return None
            else:
                raise error

    def _get_info_by_id(self, id, method):
        response = self._get_response_by_id(id, method, id_field=ApiField.ID)
        return self._convert_json_info(response.json()) if (response is not None) else None


# Base class for entities that have a parent object in the system.
class ModuleApi(ModuleApiBase):
    def __init(self, api):
        self._api = api

    def get_info_by_name(self, parent_id, name):
        return self._get_info_by_name(
            get_info_by_filters_fn=lambda module_name: self._get_info_by_filters(parent_id, module_name), name=name)

    def _get_info_by_filters(self, parent_id, filters):
        items = self.get_list(parent_id, filters)
        return _get_single_item(items)

    def get_list(self, parent_id, filters=None):
        raise NotImplementedError()

    def exists(self, parent_id, name):
        return self.get_info_by_name(parent_id, name) is not None

    def get_free_name(self, parent_id, name):
        return self._get_free_name(exist_check_fn=lambda module_name: self.exists(parent_id, module_name), name=name)

    def _get_effective_new_name(self, parent_id, name, change_name_if_conflict=False):
        return self.get_free_name(parent_id, name) if change_name_if_conflict else name


# Base class for entities that do not have a parent object in the system.
class ModuleNoParent(ModuleApiBase):
    def get_info_by_name(self, name):
        return self._get_info_by_name(get_info_by_filters_fn=self._get_info_by_filters, name=name)

    def _get_info_by_filters(self, filters):
        items = self.get_list(filters)
        return _get_single_item(items)

    def get_list(self, filters=None):
        raise NotImplementedError()

    def exists(self, name):
        return self.get_info_by_name(name) is not None

    def get_free_name(self, name):
        return self._get_free_name(exist_check_fn=lambda module_name: self.exists(module_name), name=name)

    def _get_effective_new_name(self, name, change_name_if_conflict=False):
        return self.get_free_name(name) if change_name_if_conflict else name


class CloneableModuleApi(ModuleApi):
    def _clone_api_method_name(self):
        raise NotImplementedError()

    def _clone(self, clone_type: dict, dst_workspace_id: int, dst_name: str):
        response = self._api.post(self._clone_api_method_name(), {**clone_type,
                                                                  ApiField.WORKSPACE_ID: dst_workspace_id,
                                                                  ApiField.NAME: dst_name})
        return response.json()[ApiField.TASK_ID]

    def clone(self, id, dst_workspace_id, dst_name):
        return self._clone({ApiField.ID: id}, dst_workspace_id, dst_name)

    def clone_by_shared_link(self, shared_link, dst_workspace_id, dst_name):
        return self._clone({ApiField.SHARED_LINK: shared_link}, dst_workspace_id, dst_name)

    def clone_from_explore(self, explore_path, dst_workspace_id, dst_name):
        return self._clone({ApiField.EXPLORE_PATH: explore_path}, dst_workspace_id, dst_name)

    def get_or_clone_from_explore(self, explore_path, dst_workspace_id, dst_name):
        if not self.exists(dst_workspace_id, dst_name):
            task_id = self.clone_from_explore(explore_path, dst_workspace_id, dst_name)
            self._api.task.wait(task_id, self._api.task.Status.FINISHED)
        item = self.get_info_by_name(dst_workspace_id, dst_name)
        return item


class ModuleWithStatus:
    def get_status(self, id):
        raise NotImplementedError()

    def raise_for_status(self, status):
        raise NotImplementedError()


class WaitingTimeExceeded(Exception):
    pass


class UpdateableModule(_JsonConvertibleModule):
    def __init__(self, api):
        self._api = api

    def _get_update_method(self):
        raise NotImplementedError()

    def update(self, id, name=None, description=None):
        if name is None and description is None:
            raise ValueError("\'name\' or \'description\' or both have to be specified")

        body = {ApiField.ID: id}
        if name is not None:
            body[ApiField.NAME] = name
        if description is not None:
            body[ApiField.DESCRIPTION] = description

        response = self._api.post(self._get_update_method(), body)
        return self._convert_json_info(response.json())


class RemoveableModuleApi(ModuleApi):
    def _remove_api_method_name(self):
        raise NotImplementedError()

    def remove(self, id):
        self._api.post(self._remove_api_method_name(), {ApiField.ID: id})

    def remove_batch(self, ids, progress_cb=None):
        for id in ids:
            self.remove(id)
            if progress_cb is not None:
                progress_cb(1)


class RemoveableBulkModuleApi(ModuleApi):
    def _remove_batch_api_method_name(self):
        raise NotImplementedError()

    def _remove_batch_field_name(self):
        raise NotImplementedError()

    def remove_batch(self, ids, progress_cb=None):
        for ids_batch in batched(ids):
            self._api.post(self._remove_batch_api_method_name(), {self._remove_batch_field_name(): ids_batch})
            if progress_cb is not None:
                progress_cb(len(ids_batch))

    def remove(self, id):
        self.remove_batch([id])
