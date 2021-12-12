# coding: utf-8

from supervisely_lib.api.module_api import ApiField
from supervisely_lib.api.entity_annotation.tag_api import TagApi


class VideoTagApi(TagApi):
    _entity_id_field = ApiField.VIDEO_ID
    _method_bulk_add = 'videos.tags.bulk.add'

    def remove_from_video(self, tag_id):
        self._api.post('videos.tags.remove', {ApiField.ID: tag_id})

    def update_frame_range(self, tag_id, frame_range):
        self._api.post('videos.tags.update', {ApiField.ID: tag_id, ApiField.FRAME_RANGE: frame_range})

    def update_value(self, tag_id, tag_value):
        self._api.post('videos.tags.update-value', {ApiField.ID: tag_id, ApiField.VALUE: tag_value})

    def add_tag(self, project_meta_tag_id, video_id, value=None, frame_range=None):
        request_data = {
            ApiField.TAG_ID: project_meta_tag_id,
            ApiField.VIDEO_ID: video_id
        }
        if value:
            request_data[ApiField.VALUE] = value
        if frame_range:
            request_data[ApiField.FRAME_RANGE] = frame_range

        self._api.post('videos.tags.add', request_data)


"""
    import supervisely_lib as sly

    app = sly.AppService()
    api = app.public_api

    video_info = api.video.get_info_by_id(1114874)  # get video_info_by_id 

    test_tag_id = video_info.tags[0]['id'] 
    api.video.tag.remove_from_video(test_tag_id)  # remove tag example
    api.video.tag.update_value(test_tag_id, 33221)  # update tag value
    api.video.tag.update_frame_range(test_tag_id, [0, 15])  # update tag frame range
"""
