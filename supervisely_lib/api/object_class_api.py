# coding: utf-8

from supervisely_lib.api.module_api import ModuleApi
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class ObjectClassApi(ModuleApi):
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.SHAPE,
                ApiField.COLOR,
                ApiField.SETTINGS,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT
                ]

    @staticmethod
    def info_tuple_name():
        return 'ObjectClassInfo'

    def get_list(self, project_id, filters=None):
        '''
        :param project_id: int
        :param filters: list
        :return: List the object classes from the given project
        '''
        return self.get_list_all_pages('advanced.object_classes.list',  {ApiField.PROJECT_ID: project_id, "filter": filters or []})

    def get_name_to_id_map(self, project_id):
        '''
        :param project_id: int
        :return: dictionary object class name -> object class id
        '''
        objects_infos = self.get_list(project_id)
        return {object_info.name: object_info.id for object_info in objects_infos}

    # def _object_classes_to_json(self, object_classes: KeyIndexedCollection, objclasses_name_id_map=None, project_id=None):
    #     pass #@TODO: implement
    #     # if objclasses_name_id_map is None and project_id is None:
    #     #     raise RuntimeError("Impossible to get ids for projectTags")
    #     # if objclasses_name_id_map is None:
    #     #     objclasses_name_id_map = self.get_name_to_id_map(project_id)
    #     # tags_json = []
    #     # for tag in tags:
    #     #     tag_json = tag.to_json()
    #     #     tag_json[ApiField.TAG_ID] = tag_name_id_map[tag.name]
    #     #     tags_json.append(tag_json)
    #     # return tags_json
    # 
    # def append_to_video(self, video_id, tags: KeyIndexedCollection, key_id_map: KeyIdMap = None):
    #     if len(tags) == 0:
    #         return []
    #     video_info = self._api.video.get_info_by_id(video_id)
    #     tags_json = self._tags_to_json(tags, project_id=video_info.project_id)
    #     ids = self.append_to_video_json(video_id, tags_json)
    #     KeyIdMap.add_tags_to(key_id_map, [tag.key() for tag in tags], ids)
    #     return ids
    # 
    # def append_to_video_json(self, video_id, tags_json):
    #     if len(tags_json) == 0:
    #         return
    #     response = self._api.post('videos.tags.bulk.add', {ApiField.VIDEO_ID: video_id, ApiField.TAGS: tags_json})
    #     ids = [obj[ApiField.ID] for obj in response.json()]
    #     return ids


