# coding: utf-8

# docs
from typing import Dict

from supervisely._utils import batched
from supervisely.api.module_api import ApiField
from supervisely.api.pointcloud.pointcloud_api import PointcloudApi
from supervisely.api.pointcloud.pointcloud_episode_annotation_api import (
    PointcloudEpisodeAnnotationAPI,
)
from supervisely.api.pointcloud.pointcloud_episode_object_api import (
    PointcloudEpisodeObjectApi,
)


class PointcloudEpisodeApi(PointcloudApi):
    """

    API for working with :class:`PointcloudEpisodes<supervisely.pointcloud_episodes.pointcloud_episodes>`.
    :class:`PointcloudEpisodeApi<PointcloudEpisodeApi>` object is immutable.
    Inherits from :class:`PointcloudApi<supervisely.api.pointcloud.PointcloudApi>`.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        pcd_epsodes_id = 19373295
        pcd_epsodes_info = api.pointcloud_episode.get_info_by_id(pcd_epsodes_id) # api usage example
    """

    def __init__(self, api):
        super().__init__(api)
        self.annotation = PointcloudEpisodeAnnotationAPI(api)
        self.object = PointcloudEpisodeObjectApi(api)
        self.tag = None

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        if res.meta is not None:
            return res._replace(frame=res.meta[ApiField.FRAME])
        else:
            raise RuntimeError(
                "Error with point cloud meta or API version. Please, contact support"
            )

    def get_frame_name_map(self, dataset_id: int) -> Dict:
        """
        Get a dictionary with frame_id and name of pointcloud by dataset id.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
        :type dataset_id: int
        :return: Dict with frame_id and name of pointcloud.
        :rtype: Dict

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 62664
            frame_to_name_map = api.pointcloud_episode.get_frame_name_map(dataset_id)
            print(frame_to_name_map)

            # Output:
            # {0: '001', 1: '002'}
        """

        pointclouds = self.get_list(dataset_id)

        frame_index_to_pcl_name = {}
        if len(pointclouds) > 0 and pointclouds[0].frame is None:
            pointclouds_names = sorted([x.name for x in pointclouds])
            for frame_index, pcl_name in enumerate(pointclouds_names):
                frame_index_to_pcl_name[frame_index] = pcl_name

        else:
            frame_index_to_pcl_name = {x.frame: x.name for x in pointclouds}

        return frame_index_to_pcl_name

    def notify_progress(
        self,
        track_id: int,
        dataset_id: int,
        pcd_ids: list,
        current: int,
        total: int,
    ):
        """
        Send message to the Annotation Tool and return info if tracking was stopped

        :param track_id: int
        :param dataset_id: int
        :param pcd_ids: list
        :param current: int
        :param total: int
        :return: str
        """

        response = self._api.post(
            "point-clouds.episodes.notify-annotation-tool",
            {
                "type": "point-cloud-episodes:fetch-figures-in-range",
                "data": {
                    ApiField.TRACK_ID: track_id,
                    ApiField.DATASET_ID: dataset_id,
                    ApiField.POINTCLOUD_IDS: pcd_ids,
                    ApiField.PROGRESS: {ApiField.CURRENT: current, ApiField.TOTAL: total},
                },
            },
        )
        return response.json()[ApiField.STOPPED]

    def get_max_frame_idx(self, dataset_id: int) -> int:
        """
        Get max frame index for episode by dataset id.
        This method is useful for uploading pointclouds to the episode in parts.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
        :type dataset_id: int
        :return: Max frame index.
        :rtype: int

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 62664
            max_frame = api.pointcloud_episode.get_max_frame(dataset_id)
            print(max_frame)

            # Output:
            # 1
        """

        pointclouds = self.get_list(dataset_id)
        frames = [x.frame for x in pointclouds]
        if len(frames) == 0:
            return None
        max_frame = max(frames)
        return max_frame

    def _upload_bulk_add(
        self,
        func_item_to_kv,
        dataset_id,
        names,
        items,
        metas=None,
        progress_cb=None,
    ):
        if metas is None:
            max_frame = self.get_max_frame_idx(dataset_id)
            if max_frame is None:
                max_frame = range(len(items))
            else:
                max_frame = range(max_frame + 1, max_frame + 1 + len(items))
            metas = [{ApiField.FRAME: i} for i in max_frame]
        else:
            if len(metas) != len(items):
                raise RuntimeError(
                    'Can not match "metas" and "items" lists, len(metas) != len(items)'
                )

            missing_frame_indices = [
                idx for idx, meta in enumerate(metas) if ApiField.FRAME not in meta
            ]
            if len(missing_frame_indices) == len(metas):
                raise RuntimeError("No 'frame' key found in all 'metas'.")
            elif len(missing_frame_indices) > 0:
                missing_frame_names = [names[idx] for idx in missing_frame_indices]
                raise RuntimeError(
                    f"No 'frame' key found in 'metas' for names {missing_frame_names}."
                )

        results = []
        if len(names) == 0:
            return results
        if len(names) != len(items):
            raise RuntimeError('Can not match "names" and "items" lists, len(names) != len(items)')

        for batch in batched(list(zip(names, items, metas))):
            images = []
            for name, item, meta in batch:
                item_tuple = func_item_to_kv(item)
                images.append(
                    {
                        ApiField.NAME: name,
                        item_tuple[0]: item_tuple[1],
                        ApiField.META: meta,
                    }
                )
            response = self._api.post(
                "point-clouds.bulk.add",
                {ApiField.DATASET_ID: dataset_id, ApiField.POINTCLOUDS: images},
            )
            if progress_cb is not None:
                progress_cb(len(images))

            results.extend([self._convert_json_info(item) for item in response.json()])
        name_to_res = {img_info.name: img_info for img_info in results}
        ordered_results = [name_to_res[name] for name in names]

        return ordered_results
