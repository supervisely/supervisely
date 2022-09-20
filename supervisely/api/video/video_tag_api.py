# coding: utf-8

# docs
from typing import List, Optional, Union
from supervisely.annotation.tag_meta import TagMeta
from supervisely.video_annotation.video_tag import VideoTag

from supervisely.api.module_api import ApiField
from supervisely.api.entity_annotation.tag_api import TagApi


class VideoTagApi(TagApi):
    _entity_id_field = ApiField.VIDEO_ID
    _method_bulk_add = "videos.tags.bulk.add"

    def remove_from_video(self, tag_id: int) -> None:
        """
        Remove tag from video.

        :param tag_id: VideoTag ID in Supervisely.
        :type tag_id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.video.tag.remove_from_video(video_tag_id)

        """
        self._api.post("videos.tags.remove", {ApiField.ID: tag_id})

    def update_frame_range(self, tag_id: int, frame_range: List[int]) -> None:
        """
        Update VideoTag frame range in video.

        :param tag_id: VideoTag ID in Supervisely.
        :type tag_id: int
        :param frame_range: New VideoTag frame range.
        :type frame_range: List[int]
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            new_frame_range = [5, 10]
            api.video.tag.update_frame_range(video_tag_id, new_frame_range)
        """
        self._api.post(
            "videos.tags.update", {ApiField.ID: tag_id, ApiField.FRAME_RANGE: frame_range}
        )

    def update_value(self, tag_id: int, tag_value: Union[str, int]) -> None:
        """
        Update VideoTag value.

        :param tag_id: VideoTag ID in Supervisely.
        :type tag_id: int
        :param tag_value: New VideoTag value.
        :type tag_value: str or int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.video.tag.update_value(video_tag_id, 'new_tag_value')
        """
        self._api.post("videos.tags.update-value", {ApiField.ID: tag_id, ApiField.VALUE: tag_value})

    def add_tag(
        self,
        project_meta_tag_id: int,
        video_id: int,
        value: Optional[Union[str, int]] = None,
        frame_range: Optional[List[int]] = None,
    ) -> int:
        """
        Add VideoTag to video.

        :param project_meta_tag_id: TagMeta ID in Supervisely.
        :type project_meta_tag_id: int
        :param video_id: Video ID in Supervidely.
        :type video_id: int
        :param value: New VideoTag value.
        :type value: str or int
        :param frame_range: New VideoTag frame range.
        :type frame_range: List[int]
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            new_frame_range = [5, 10]
            api.video.tag.add_tag(project_meta_tag_id, video_id, 'tag_value', frame_range)
        """
        request_data = {ApiField.TAG_ID: project_meta_tag_id, ApiField.VIDEO_ID: video_id}
        if value:
            request_data[ApiField.VALUE] = value
        if frame_range:
            request_data[ApiField.FRAME_RANGE] = frame_range

        resp = self._api.post("videos.tags.add", request_data)
        # {'imageId': 3267369, 'tagId': 368985, 'id': 2296671}
        return resp.json()["id"]

    def add(self, video_id: int, tag: VideoTag, update_id_inplace=True) -> int:
        from supervisely.project.project_meta import ProjectMeta

        if tag.meta.sly_id is None:
            if update_id_inplace is True:
                video_info = self._api.video.get_info_by_id(video_id)
                meta_json = self._api.project.get_meta(video_info.project_id)
                meta = ProjectMeta.from_json(meta_json)
                server_tag_meta = meta.get_tag_meta(tag.meta.name)
                if server_tag_meta is None:
                    raise KeyError(
                        f"Tag with name {tag.meta.name} not found in project with id {video_info.project_id}"
                    )
                tag.meta._set_id(server_tag_meta.sly_id)
            else:
                raise ValueError("tag_meta.sly_id is None, get updated project meta from server")

        tag_id = self.add_tag(tag.meta.sly_id, video_id, tag.value, tag.frame_range)
        if update_id_inplace is True:
            tag._set_id(tag_id)
        return tag_id
