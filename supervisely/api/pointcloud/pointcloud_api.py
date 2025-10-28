# coding: utf-8

# docs
import asyncio
import os
from collections import defaultdict
from typing import (
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Union,
)

import aiofiles
from requests import Response
from requests_toolbelt import MultipartEncoder
from tqdm import tqdm

from supervisely._utils import batched, generate_free_name
from supervisely.api.module_api import ApiField, RemoveableBulkModuleApi
from supervisely.api.pointcloud.pointcloud_annotation_api import PointcloudAnnotationAPI
from supervisely.api.pointcloud.pointcloud_figure_api import PointcloudFigureApi
from supervisely.api.pointcloud.pointcloud_object_api import PointcloudObjectApi
from supervisely.api.pointcloud.pointcloud_tag_api import PointcloudTagApi
from supervisely.io.fs import (
    ensure_base_path,
    get_file_hash,
    get_file_hash_async,
    get_file_name_with_ext,
    list_files,
    list_files_recursively,
)
from supervisely.pointcloud.pointcloud import is_valid_format
from supervisely.sly_logger import logger
from supervisely.imaging import image as sly_image


class PointcloudInfo(NamedTuple):
    """
    Object with :class:`Pointcloud<supervisely.pointcloud.pointcloud>` parameters from Supervisely.

    :Example:

    .. code-block:: python

        PointcloudInfo(
            id=19373403,
            frame=None,
            description='',
            name='000063.pcd',
            team_id=435,
            workspace_id=687,
            project_id=17231,
            dataset_id=55875,
            link=None,
            hash='7EcJCyhq15V4NnZ8oiPrKQckmXXypO4saqFN7kgH08Y=',
            path_original='/h5unms4-public/point_clouds/Z/h/bc/roHZP5nP2.pcd',
            cloud_mime='image/pcd',
            figures_count=4,
            objects_count=4,
            tags=[],
            meta={},
            created_at='2023-02-07T19:36:44.897Z',
            updated_at='2023-02-07T19:36:44.897Z'
        )
    """

    #: :class:`int`: Point cloud ID in Supervisely.
    id: int

    #: :class:`int`: Number of frame in the point cloud
    frame: int

    #: :class:`str`: Point cloud description.
    description: str

    #: :class:`str`: Point cloud filename.
    name: str

    #: :class:`int`: :class:`TeamApi<supervisely.api.team_api.TeamApi>` ID in Supervisely.
    team_id: int

    #: :class:`int`: :class:`WorkspaceApi<supervisely.api.workspace_api.WorkspaceApi>` ID in Supervisely.
    workspace_id: int

    #: :class:`int`: :class:`Project<supervisely.project.project.Project>` ID in Supervisely.
    project_id: int

    #: :class:`int`: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
    dataset_id: int

    #: :class:`str`: Link to point cloud.
    link: str

    #: :class:`str`: Point cloud hash obtained by base64(sha256(file_content)).
    #: Use hash for files that are expected to be stored at Supervisely or your deployed agent.
    hash: str

    #: :class:`str`: Relative storage URL to point cloud. e.g.
    #: "/h5un6l2bnaz1vms4-public/pointclouds/Z/d/HD/lfgipl...NXrg5vz.mp4".
    path_original: str

    #: :class:`str`: MIME type of the point cloud.
    cloud_mime: str

    #: :class:`int`: Number of PointcloudFigure objects in the point cloud
    figures_count: int

    #: :class:`int`: Number of PointcloudObject objects in the point cloud
    objects_count: int

    #: :class:`list`: Pointcloud :class:`PointcloudTag<supervisely.pointcloud_annotation.pointcloud_tag.PointcloudTag>` list.
    tags: list

    #: :class:`dict`: A dictionary containing point cloud metadata.
    meta: dict

    #: :class:`str`: Point cloud creation time. e.g. "2019-02-22T14:59:53.381Z".
    created_at: str

    #: :class:`str`: Time of last point cloud update. e.g. "2019-02-22T14:59:53.381Z".
    updated_at: str


class PointcloudApi(RemoveableBulkModuleApi):
    """
    API for working with :class:`Pointcloud<supervisely.pointcloud.pointcloud>`.
    :class:`PointcloudApi<PointcloudApi>` object is immutable.

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

        pcd_id = 19618654
        pcd_info = api.pointcloud.get_info_by_id(pcd_id) # api usage example
    """

    def __init__(self, api):
        """
        :param api: Api class object
        """
        super().__init__(api)
        self.annotation = PointcloudAnnotationAPI(api)
        self.object = PointcloudObjectApi(api)
        self.figure = PointcloudFigureApi(api)
        self.tag = PointcloudTagApi(api)

    @staticmethod
    def info_sequence():
        """
        Get list of all :class:`PointcloudInfo<PointcloudInfo>` field names.

        :return: List of :class:`PointcloudInfo<PointcloudInfo>` field names.`
        :rtype: :class:`list`
        """

        return [
            ApiField.ID,
            ApiField.FRAME,
            ApiField.DESCRIPTION,
            ApiField.NAME,
            ApiField.TEAM_ID,
            ApiField.WORKSPACE_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.LINK,
            ApiField.HASH,
            ApiField.PATH_ORIGINAL,
            # ApiField.PREVIEW,
            ApiField.CLOUD_MIME,
            ApiField.FIGURES_COUNT,
            ApiField.ANN_OBJECTS_COUNT,
            ApiField.TAGS,
            ApiField.META,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        Get string name of :class:`PointcloudInfo<PointcloudInfo>` NamedTuple.

        :return: NamedTuple name.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            tuple_name = api.pointcloud.info_tuple_name()
            print(tuple_name) # PointCloudInfo
        """

        return "PointCloudInfo"

    def _convert_json_info(self, info: Dict, skip_missing: Optional[bool] = True):
        res = super(PointcloudApi, self)._convert_json_info(info, skip_missing=skip_missing)
        return PointcloudInfo(**res._asdict())

    def get_list(
        self,
        dataset_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
    ) -> List[PointcloudInfo]:
        """
        Get list of information about all point cloud for a given dataset ID.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
        :type dataset_id: int
        :param filters: List of parameters to sort output Pointclouds. See: https://api.docs.supervisely.com/#tag/Point-Clouds/paths/~1point-clouds.list/get
        :type filters: List[Dict[str, str]], optional
        :return: List of the point clouds objects from the dataset with given id.
        :rtype: :class:`List[PointcloudInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 62664
            pcd_infos = api.pointcloud_episode.get_list(dataset_id)
            print(pcd_infos)
            # Output: [PointcloudInfo(...), PointcloudInfo(...)]

            id_list = [19618654, 19618657, 19618660]
            filtered_pointcloud_infos = api.pointcloud.get_list(dataset_id, filters=[{'field': 'id', 'operator': 'in', 'value': id_list}])
            print(filtered_pointcloud_infos)
            # Output:
            # [PointcloudInfo(id=19618654, ...), PointcloudInfo(id=19618657, ...), PointcloudInfo(id=19618660, ...)]
        """

        return self.get_list_all_pages(
            "point-clouds.list",
            {
                ApiField.DATASET_ID: dataset_id,
                ApiField.FILTER: filters or [],
            },
        )

    def get_info_by_id(self, id: int) -> PointcloudInfo:
        """
        Get point cloud information by ID in PointcloudInfo<PointcloudInfo> format.

        :param id: Point cloud ID in Supervisely.
        :type id: int
        :param raise_error: Return an error if the point cloud info was not received.
        :type raise_error: bool
        :return: Information about point cloud. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`PointcloudInfo`

        :Usage example:

         .. code-block:: python

            import supervisely as sly


            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pcd_id = 19373403
            pcd_info = api.pointcloud.get_info_by_id(pcd_id)
            print(pcd_info)

            # Output:
            # PointcloudInfo(
            #     id=19373403,
            #     frame=None,
            #     description='',
            #     name='000063.pcd',
            #     team_id=435,
            #     workspace_id=687,
            #     project_id=17231,
            #     dataset_id=55875,
            #     link=None,
            #     hash='7EcJCyhq15V4NnZ8oiPrKQckmXXypO4saqFN7kgH08Y=',
            #     path_original='/h5unms4-public/point_clouds/Z/h/bc/roHZP5nP2.pcd',
            #     cloud_mime='image/pcd',
            #     figures_count=4,
            #     objects_count=4,
            #     tags=[],
            #     meta={},
            #     created_at='2023-02-07T19:36:44.897Z',
            #     updated_at='2023-02-07T19:36:44.897Z'
            # )
        """
        return self._get_info_by_id(id, "point-clouds.info")

    def _download(self, id: int, is_stream: Optional[bool] = False):
        """
        :param id: int
        :param is_stream: bool
        :return: Response object containing pointcloud object with given id
        """
        response = self._api.post(
            "point-clouds.download",
            {ApiField.ID: id},
            stream=is_stream,
        )
        return response

    def download_path(self, id: int, path: str) -> None:
        """
        Download point cloud with given id on the given path.

        :param id: Point cloud ID in Supervisely.
        :type id: int
        :param path: Local save path for point cloud.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            storage_dir = sly.app.get_data_dir()
            pcd_id = 19373403
            pcd_info = api.pointcloud.get_info_by_id(pcd_id)
            save_path = os.path.join(storage_dir, pcd_info.name)

            api.pointcloud.download_path(pcd_id, save_path)
            print(os.listdir(storage_dir))

            # Output: ['000063.pcd']
        """

        response = self._download(id, is_stream=True)
        ensure_base_path(path)
        with open(path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

    def get_list_related_images(self, id: int) -> List:
        """
        Get information about related context images.

        :param id: Point cloud ID in Supervisely.
        :type id: int
        :return: List of dictionaries with informations about related images
        :rtype: List
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pcd_id = 19373403
            img_infos = api.pointcloud.get_list_related_images(pcd_id)
            img_info = img_infos[0]
            print(img_info)

            # Output:
            # {
            #     'pathOriginal': '/h5un6qgms4-public/images/original/S/j/hJ/PwMg.png',
            #     'id': 473302,
            #     'entityId': 19373403,
            #     'createdAt': '2023-01-09T08:50:33.225Z',
            #     'updatedAt': '2023-01-09T08:50:33.225Z',
            #     'meta': {
            #         'deviceId': 'cam_2'},
            #         'fileMeta': {'mime': 'image/png',
            #         'size': 893783,
            #         'width': 1224,
            #         'height': 370
            #     },
            #     'hash': 'vxA+emfDNUkFP9P6oitMB5Q0rMlnskmV2jvcf47OjGU=',
            #     'link': None,
            #     'preview': '/previews/q/ext:jpeg/resize:fill:50:0:0/q:50/plain/h5ad-public/images/original/S/j/hJ/PwMg.png',
            #     'fullStorageUrl': 'https://app.supervisely.com/hs4-public/images/original/S/j/hJ/PwMg.png',
            #     'name': 'img00'
            # }
        """

        dataset_id = self.get_info_by_id(id).dataset_id
        filters = [{"field": ApiField.ENTITY_ID, "operator": "=", "value": id}]
        return self.get_list_all_pages(
            "point-clouds.images.list",
            {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters},
            convert_json_info_cb=lambda x: x,
        )

    def get_list_related_images_batch(self, dataset_id: int, ids: List[int]) -> List:
        filters = [{"field": ApiField.ENTITY_ID, "operator": "in", "value": ids}]
        return self.get_list_all_pages(
            "point-clouds.images.list",
            {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters},
            convert_json_info_cb=lambda x: x,
        )

    def download_related_image(self, id: int, path: str = None) -> Response:
        """
        Download a related context image from Supervisely to local directory by image id.

        :param id: Related context imgage ID in Supervisely.
        :type id: int
        :param path: Local save path for point cloud.
        :type path: str
        :return: List of dictionaries with informations about related images
        :rtype: List
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            save_path = "src/output/img_0.png"
            img_info = api.pointcloud.get_list_related_images(pcd_info.id)[0]
            api.pointcloud.download_related_image(img_info["id"], save_path)
            print(f"Context image has been successfully downloaded to '{save_path}'")

        # Output:
        # Context image has been successfully downloaded to 'src/output/img_0.png'
        """

        response = self._api.post(
            "point-clouds.images.download",
            {ApiField.ID: id},
            stream=True,
        )

        if path:
            ensure_base_path(path)
            with open(path, "wb") as fd:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    fd.write(chunk)
            return response
        else:
            related_image = sly_image.read_bytes(response.content, False)
            return related_image

    # @TODO: copypaste from video_api
    def upload_hash(
        self,
        dataset_id: int,
        name: str,
        hash: str,
        meta: Optional[Dict] = None,
    ) -> PointcloudInfo:
        """
        Upload Pointcloud from given hash to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Point cloud name with extension.
        :type name: str
        :param hash: Point cloud hash.
        :type hash: str
        :param meta: Point cloud metadata.
        :type meta: dict, optional
        :return: Information about point cloud. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`PointcloudInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_dataset_id = 62693

            src_pointcloud_id = 19618685
            pcd_info = api.pointcloud.get_info_by_id(src_pointcloud_id)
            hash, name, meta = pcd_info.hash, pcd_info.name, pcd_info.meta

            new_pcd_info = api.pointcloud.upload_hash(dst_dataset_id.id, name, hash, meta)
            print(new_pcd_info)

            # Output:
            # PointcloudInfo(
            #     id=19619507,
            #     frame=None,
            #     description='',
            #     name='0000000031.pcd',
            #     team_id=None,
            #     workspace_id=None,
            #     project_id=None,
            #     dataset_id=62694,
            #     link=None,
            #     hash='5w69Vv1i6JrqhU0Lw1UJAJFGPVWUzDG7O3f4QSwRfmE=',
            #     path_original='/j8a9qgms4-public/point_clouds/I/3/6U/L7YBY.pcd',
            #     cloud_mime='image/pcd',
            #     figures_count=None,
            #     objects_count=None,
            #     tags=None,
            #     meta={'frame': 31},
            #     created_at='2023-04-05T10:59:44.656Z',
            #     updated_at='2023-04-05T10:59:44.656Z'
            # )
        """

        meta = {} if meta is None else meta
        return self.upload_hashes(dataset_id, [name], [hash], [meta])[0]

    # @TODO: copypaste from video_api
    def upload_hashes(
        self,
        dataset_id: int,
        names: List[str],
        hashes: List[str],
        metas: Optional[List[Dict]] = None,
        progress_cb: Optional[Callable] = None,
    ) -> List[PointcloudInfo]:
        """
        Upload point clouds from given hashes to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Point cloud name with extension.
        :type names: List[str]
        :param hashes: Point cloud hash.
        :type hashes: List[str]
        :param metas: Point cloud metadata.
        :type metas: Optional[List[Dict]], optional
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: Progress, optional
        :return: List of informations about Pointclouds. See :class:`info_sequence<info_sequence>`
        :rtype: List[:class:`PointcloudInfo`]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_dataset_id = 62664
            dst_dataset_id = 62690

            src_pcd_infos = api.pointcloud.get_list(src_dataset_id)

            names = [pcd.name for pcd in src_pcd_infos[:4]]
            hashes = [pcd.hash for pcd in src_pcd_infos[:4]]
            metas = [pcd.meta for pcd in src_pcd_infos[:4]]

            dst_pcd_infos = api.pointcloud.get_list(dst_dataset_id)
            print(f"{len(dst_pcd_infos)} pointcloud before upload.")
            # Output:
            # 0 pointcloud before upload.

            new_pcd_infos = api.pointcloud.upload_hashes(dst_dataset_id, names, hashes, metas)
            print(f"{len(new_pcd_infos)} pointcloud after upload.")
            # Output:
            # 4 pointcloud after upload.
        """

        return self._upload_bulk_add(
            lambda item: (ApiField.HASH, item), dataset_id, names, hashes, metas, progress_cb
        )

    # @TODO: copypaste from video_api
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
            metas = [{}] * len(items)

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
                        ApiField.META: meta if meta is not None else {},
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

    def upload_related_image(self, path: str) -> str:
        """
        Upload an image to the Supervisely. It generates us a hash for image.

        :param path: Image path.
        :type path: str
        :return: Hash for image. See :class:`info_sequence<info_sequence>`
        :rtype: str
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_file = src/input/img/000000.png"
            img_hash = api.pointcloud.upload_related_image(img_file)
            print(img_hash)

            # Output:
            # +R6dFy8nMEq6k82vHLxuakpqVBmyTTPj5hXdPfjAv/c=
        """

        return self.upload_related_images([path])[0]

    def upload_related_images(
        self,
        paths: List[str],
        progress_cb: Optional[Callable] = None,
    ) -> List[str]:
        """
        Upload a batch of related images to the Supervisely. It generates us a hashes for images.

        :param paths: Images pathes.
        :type paths: List[str]
        :return: List of hashes for images. See :class:`info_sequence<info_sequence>`
        :rtype: List[str]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_paths = ["src/input/img/000001.png", "src/input/img/000002.png"]
            img_hashes = api.pointcloud.upload_related_images(img_paths)

            # Output:
            # [+R6dFy8nMEq6k82vHLxuakpqVBmyTTPjdfGdPfjAv/c=, +hfjbufnbkLhJb32vHLxuakpqVBmyTTPj5hXdPfhhj1c]
        """

        def path_to_bytes_stream(path):
            return open(path, "rb")

        return self._upload_data_bulk(path_to_bytes_stream, get_file_hash, paths, progress_cb)

    def add_related_images(
        self,
        images_json: List[Dict],
        camera_names: List[str] = None,
    ) -> Dict:
        """
        Attach images to point cloud.

        :param images_json: List of dictionaries with dataset id, image name, hash and meta.
        :type images_json: List[Dict]
        :param camera_names: List of camera informations.
        :type camera_names: List[Dict]
        :return: Response object
        :rtype: Dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_paths = ["src/input/img/000001.png", "src/input/img/000002.png"]
            cam_paths = ["src/input/cam_info/000001.json", "src/input/cam_info/000002.json"]

            img_hashes = api.pointcloud.upload_related_images(img_paths)
            img_infos = []
            for i, cam_info_file in enumerate(cam_paths):
                # reading cam_info
                with open(cam_info_file, "r") as f:
                    cam_info = json.load(f)
                img_info = {
                    "entityId": pcd_infos[i].id,
                    "name": f"img_{i}.png",
                    "hash": img_hashes[i],
                    "meta": cam_info,
                }
                img_infos.append(img_info)
            api.pointcloud.add_related_images(img_infos)
        """

        if camera_names is not None:
            if len(camera_names) != len(images_json):
                ValueError("camera_names length must be equal to images_json length.")
            for img_ind, camera_name in enumerate(camera_names):
                images_json[img_ind][ApiField.META]["deviceId"] = camera_name
        response = self._api.post("point-clouds.images.add", {ApiField.IMAGES: images_json})
        return response.json()

    def upload_path(
        self,
        dataset_id: int,
        name: str,
        path: str,
        meta: Optional[Dict] = None,
    ) -> PointcloudInfo:
        """
        Upload point cloud with given path to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Point cloud name with extension.
        :type name: str
        :param path: Path to point cloud.
        :type path: str
        :param meta: Dictionary with metadata for point cloud.
        :type meta: Optional[Dict]
        :return: Information about point cloud
        :rtype: PointcloudInfo
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pcd_file = "src/input/pcd/000000.pcd"
            pcd_info = api.pointcloud.upload_path(dataset.id, name="pcd_0.pcd", path=pcd_file)
            print(f'Point cloud "{pcd_info.name}" uploaded to Supervisely with ID:{pcd_info.id}')

            # Output:
            # Point cloud "pcd_0.pcd" uploaded to Supervisely with ID:19618685
        """

        metas = None if meta is None else [meta]
        return self.upload_paths(dataset_id, [name], [path], metas=metas)[0]

    def upload_paths(
        self,
        dataset_id: int,
        names: List[str],
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
    ) -> List[PointcloudInfo]:
        """
        Upload point clouds with given paths to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Point clouds names with extension.
        :type names: List[str]
        :param paths: Paths to point clouds.
        :type paths: List[str]
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: Progress, optional
        :param metas: List of dictionary with metadata for point cloud.
        :type metas: Optional[List[Dict]]
        :return: List of informations about point clouds
        :rtype: List[PointcloudInfo]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            paths = ["src/input/pcd/000001.pcd", "src/input/pcd/000002.pcd"]
            pcd_infos = api.pointcloud.upload_paths(dataset.id, names=["pcd_1.pcd", "pcd_2.pcd"], paths=paths)
            print(f'Point clouds uploaded to Supervisely with IDs: {[pcd_info.id for pcd_info in pcd_infos]}')

            # Output:
            # Point clouds uploaded to Supervisely with IDs: [19618685, 19618686]
        """

        def path_to_bytes_stream(path):
            return open(path, "rb")

        hashes = self._upload_data_bulk(path_to_bytes_stream, get_file_hash, paths, progress_cb)
        return self.upload_hashes(dataset_id, names, hashes, metas=metas)

    def check_existing_hashes(self, hashes: List[str]) -> List[str]:
        """
        Check if point clouds with given hashes are exist.

        :param paths: Point clouds hashes to check.
        :type paths: List[str]
        :return: List of point clouds hashes that are exist.
        :rtype: List[str]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pointcloud_id = 19618685
            pcd_info = api.pointcloud.get_info_by_id(pointcloud_id)
            hash = api.pointcloud.check_existing_hashes([pcd_info.hash])
            print(hash)

            # Output:
            # ['5w69Vv1i6JrqhU0Lw1UJAJFGPhgkIhs7O3f4QSwRfmE=']
        """

        results = []
        if len(hashes) == 0:
            return results
        for hashes_batch in batched(hashes, batch_size=900):
            response = self._api.post("images.internal.hashes.list", hashes_batch)
            results.extend(response.json())
        return results

    def _upload_data_bulk(self, func_item_to_byte_stream, func_item_hash, items, progress_cb):
        hashes = []
        if len(items) == 0:
            return hashes

        hash_to_items = defaultdict(list)

        for idx, item in enumerate(items):
            item_hash = func_item_hash(item)
            hashes.append(item_hash)
            hash_to_items[item_hash].append(item)

        # count number of items for each hash
        items_number_for_hashes = {hash: len(items) for hash, items in hash_to_items.items()}

        unique_hashes = set(hashes)
        remote_hashes = self.check_existing_hashes(list(unique_hashes))
        new_hashes = unique_hashes - set(remote_hashes)

        if progress_cb is not None:
            total_remote_items = sum([items_number_for_hashes[hash] for hash in remote_hashes])
            progress_cb(total_remote_items)

        # upload only new unique images to supervisely server
        items_to_upload = [hash_to_items[hash][0] for hash in new_hashes]
        total_nem_items_list = [items_number_for_hashes[hash] for hash in new_hashes]

        for batch, numbers_batch in zip(batched(items_to_upload), batched(total_nem_items_list)):
            content_dict = {}
            for idx, item in enumerate(batch):
                content_dict["{}-file".format(idx)] = (
                    str(idx),
                    func_item_to_byte_stream(item),
                    "pcd/*",
                )
            encoder = MultipartEncoder(fields=content_dict)
            self._api.post("point-clouds.bulk.upload", encoder)

            if progress_cb is not None:
                progress_cb(sum(numbers_batch))

            for value in content_dict.values():
                from io import BufferedReader

                if isinstance(value[1], BufferedReader):
                    value[1].close()

        if not items_to_upload:
            total_unique_items = sum([items_number_for_hashes[hash] for hash in unique_hashes])
            if progress_cb is not None:
                progress_cb(len(hashes) - total_unique_items)

        return hashes

    def get_free_names(self, dataset_id: int, names: List[str]) -> List[str]:
        """
        Returns list of free names for given dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: List of names to check.
        :type names: List[str]
        :return: List of free names.
        :rtype: List[str]
        """

        pcds_in_dataset = self.get_list(dataset_id)
        used_names = {pcd_info.name for pcd_info in pcds_in_dataset}
        new_names = [
            generate_free_name(used_names, name, with_ext=True, extend_used_names=True)
            for name in names
        ]
        return new_names

    def raise_name_intersections_if_exist(
        self, dataset_id: int, names: List[str], message: str = None
    ):
        """
        Raises error if pointclouds with given names already exist in dataset.
        Default error message:
        "Pointclouds with the following names already exist in dataset [ID={dataset_id}]: {name_intersections}.
        Please, rename pointclouds and try again or set change_name_if_conflict=True to rename automatically on upload."

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: List of names to check.
        :type names: List[str]
        :param message: Error message.
        :type message: str, optional
        :return: None
        :rtype: None
        """
        pcds_in_dataset = self.get_list(dataset_id)
        used_names = {pcd_info.name for pcd_info in pcds_in_dataset}
        name_intersections = used_names.intersection(set(names))
        if message is None:
            message = f"Pointclouds with the following names already exist in dataset [ID={dataset_id}]: {name_intersections}. Please, rename pointclouds and try again or set change_name_if_conflict=True to rename automatically on upload."
        if len(name_intersections) > 0:
            raise ValueError(f"{message}")

    def upload_dir(
        self,
        dataset_id: int,
        dir_path: str,
        recursive: Optional[bool] = True,
        change_name_if_conflict: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[PointcloudInfo]:
        """
        Uploads all pointclouds with supported extensions from given directory to Supervisely.
        Optionally, uploads pointclouds from subdirectories of given directory.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param dir_path: Path to directory with pointclouds.
        :type dir_path: str
        :param recursive: If True, will upload pointclouds from subdirectories of given directory recursively. Otherwise, will upload pointclouds only from given directory.
        :type recursive: bool, optional
        :param change_name_if_conflict: If True adds suffix to the end of Pointcloud name when Dataset already contains an Pointcloud with identical name, If False and pointclouds with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :return: List of uploaded pointclouds infos
        :rtype: List[PointcloudInfo]
        """

        if os.path.isdir(dir_path) is False:
            raise ValueError(f"Path {dir_path} is not a directory or does not exist")

        if recursive:
            paths = list_files_recursively(dir_path, filter_fn=is_valid_format)
        else:
            paths = list_files(dir_path, filter_fn=is_valid_format)

        paths.sort()

        names = [get_file_name_with_ext(path) for path in paths]

        if change_name_if_conflict is True:
            names = self.get_free_names(dataset_id, names)
        else:
            self.raise_name_intersections_if_exist(dataset_id, names)

        pcds_infos = self.upload_paths(dataset_id, names, paths, progress_cb=progress_cb)
        return pcds_infos

    def upload_dirs(
        self,
        dataset_id: int,
        dir_paths: List[str],
        recursive: Optional[bool] = True,
        change_name_if_conflict: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[PointcloudInfo]:
        """
        Uploads all pointclouds with supported extensions from given directories to Supervisely.
        Optionally, uploads pointclouds from subdirectories of given directories.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param dir_paths: List of paths to directories with pointclouds.
        :type dir_paths: List[str]
        :param recursive: If True, will upload pointclouds from subdirectories of given directories recursively. Otherwise, will upload pointclouds only from given directories.
        :type recursive: bool, optional
        :param change_name_if_conflict: If True adds suffix to the end of Pointclouds name when Dataset already contains an Pointclouds with identical name, If False and pointclouds with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :return: List of uploaded pointclouds infos
        :rtype: List[Pointclouds]
        """
        if not isinstance(dir_paths, list):
            raise TypeError(f"dir_paths must be a list of strings, but got {type(dir_paths)}")

        pcds_infos = []
        for dir_path in dir_paths:
            pcds_infos.extend(
                self.upload_dir(
                    dataset_id, dir_path, recursive, change_name_if_conflict, progress_cb
                )
            )
        return pcds_infos

    async def _download_async(
        self,
        id: int,
        is_stream: bool = False,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        headers: dict = None,
        chunk_size: int = 1024 * 1024,
    ) -> AsyncGenerator:
        """
        Download Point cloud with given ID asynchronously.
        If is_stream is True, returns stream of bytes, otherwise returns response object.
        For streaming, returns tuple of chunk and hash.

        :param id: Point cloud ID in Supervisely.
        :type id: int
        :param is_stream: If True, returns stream of bytes, otherwise returns response object.
        :type is_stream: bool, optional
        :param range_start: Start byte of range for partial download.
        :type range_start: int, optional
        :param range_end: End byte of range for partial download.
        :type range_end: int, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param chunk_size: Size of chunk for partial download. Default is 1MB.
        :type chunk_size: int, optional
        :return: Stream of bytes or response object.
        :rtype: AsyncGenerator
        """
        api_method_name = "point-clouds.download"

        json_body = {ApiField.ID: id}

        if is_stream:
            async for chunk, hhash in self._api.stream_async(
                api_method_name,
                "POST",
                json_body,
                headers=headers,
                range_start=range_start,
                range_end=range_end,
                chunk_size=chunk_size,
            ):
                yield chunk, hhash
        else:
            response = await self._api.post_async(api_method_name, json_body, headers=headers)
            yield response

    async def download_path_async(
        self,
        id: int,
        path: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        headers: Optional[dict] = None,
        chunk_size: int = 1024 * 1024,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> None:
        """
        Downloads Point cloud with given ID to local path.

        :param id: Point cloud ID in Supervisely.
        :type id: int
        :param path: Local save path for Point cloud.
        :type path: str
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param range_start: Start byte of range for partial download.
        :type range_start: int, optional
        :param range_end: End byte of range for partial download.
        :type range_end: int, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param chunk_size: Size of chunk for partial download. Default is 1MB.
        :type chunk_size: int, optional
        :param check_hash: If True, checks hash of downloaded file.
                        Check is not supported for partial downloads.
                        When range is set, hash check is disabled.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import asyncio

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pcd_info = api.pointcloud.get_info_by_id(19373403)
            save_path = os.path.join("/path/to/save/", pcd_info.name)

            semaphore = asyncio.Semaphore(100)
            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(
                        api.pointcloud.download_path_async(pcd_info.id, save_path, semaphore)
                    )
        """

        if range_start is not None or range_end is not None:
            check_hash = False  # Hash check is not supported for partial downloads
            headers = headers or {}
            headers["Range"] = f"bytes={range_start or ''}-{range_end or ''}"
            logger.debug(f"Image ID: {id}. Setting Range header: {headers['Range']}")

        writing_method = "ab" if range_start not in [0, None] else "wb"

        ensure_base_path(path)
        hash_to_check = None
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        async with semaphore:
            async with aiofiles.open(path, writing_method) as fd:
                async for chunk, hhash in self._download_async(
                    id,
                    is_stream=True,
                    headers=headers,
                    range_start=range_start,
                    range_end=range_end,
                    chunk_size=chunk_size,
                ):
                    await fd.write(chunk)
                    hash_to_check = hhash
                    if progress_cb is not None and progress_cb_type == "size":
                        progress_cb(len(chunk))
                if check_hash:
                    if hash_to_check is not None:
                        downloaded_bytes_hash = await get_file_hash_async(path)
                        if hash_to_check != downloaded_bytes_hash:
                            raise RuntimeError(
                                f"Downloaded hash of point cloud with ID:{id} does not match the expected hash: {downloaded_bytes_hash} != {hash_to_check}"
                            )
                if progress_cb is not None and progress_cb_type == "number":
                    progress_cb(1)

    async def download_paths_async(
        self,
        ids: List[int],
        paths: List[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        headers: dict = None,
        chunk_size: int = 1024 * 1024,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> None:
        """
        Download Point clouds with given IDs and saves them to given local paths asynchronously.

        :param ids: List of Point cloud IDs in Supervisely.
        :type ids: :class:`List[int]`
        :param paths: Local save paths for Point clouds.
        :type paths: :class:`List[str]`
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param show_progress: If True, shows progress bar.
        :type show_progress: bool, optional
        :param chunk_size: Size of chunk for partial download. Default is 1MB.
        :type chunk_size: int, optional
        :param check_hash: If True, checks hash of downloaded file.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: Literal["number", "size"], optional
        :raises: :class:`ValueError` if len(ids) != len(paths)
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import asyncio

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            ids = [19373403, 19373404]
            paths = ["/path/to/save/000063.pcd", "/path/to/save/000064.pcd"]
            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(api.pointcloud.download_paths_async(ids, paths))
        """
        if len(ids) == 0:
            return
        if len(ids) != len(paths):
            raise ValueError('Can not match "ids" and "paths" lists, len(ids) != len(paths)')
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        tasks = []
        for img_id, img_path in zip(ids, paths):
            task = self.download_path_async(
                img_id,
                img_path,
                semaphore,
                headers=headers,
                chunk_size=chunk_size,
                check_hash=check_hash,
                progress_cb=progress_cb,
                progress_cb_type=progress_cb_type,
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def download_related_image_async(
        self,
        id: int,
        path: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        headers: dict = None,
        chunk_size: int = 1024 * 1024,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> None:
        """
        Downloads a related context image from Supervisely to local directory by image id.

        :param id: Point cloud ID in Supervisely.
        :type id: int
        :param path: Local save path for Point cloud.
        :type path: str
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param chunk_size: Size of chunk for partial download. Default is 1MB.
        :type chunk_size: int, optional
        :param check_hash: If True, checks hash of downloaded file.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import asyncio

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_info = api.pointcloud.get_list_related_images(19373403)[0]
            save_path = os.path.join("/path/to/save/", img_info.name)

            semaphore = asyncio.Semaphore(100)
            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(
                    api.pointcloud.download_related_image_async(19373403, save_path, semaphore)
                )
        """

        api_method_name = "point-clouds.images.download"

        ensure_base_path(path)
        hash_to_check = None
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        async with semaphore:
            async with aiofiles.open(path, "wb") as fd:
                async for chunk, hhash in self._api.stream_async(
                    api_method_name,
                    "POST",
                    {ApiField.ID: id},
                    headers=headers,
                    chunk_size=chunk_size,
                ):
                    await fd.write(chunk)
                    hash_to_check = hhash
                    if progress_cb is not None and progress_cb_type == "size":
                        progress_cb(len(chunk))
                if check_hash:
                    if hash_to_check is not None:
                        downloaded_bytes_hash = await get_file_hash_async(path)
                        if hash_to_check != downloaded_bytes_hash:
                            raise RuntimeError(
                                f"Downloaded hash of point cloud related image with ID:{id} does not match the expected hash: {downloaded_bytes_hash} != {hash_to_check}"
                            )
                if progress_cb is not None and progress_cb_type == "number":
                    progress_cb(1)

    async def download_related_images_async(
        self,
        ids: List[int],
        paths: List[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        headers: dict = None,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> None:
        """
        Downloads a related context image from Supervisely to local directory by image id.

        :param ids: Related context imgage IDs in Supervisely.
        :type ids: int
        :param paths: Local save paths for Point clouds.
        :type paths: str
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param check_hash: If True, checks hash of downloaded file.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

            .. code-block:: python

                import supervisely as sly
                import asyncio

                os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
                os.environ['API_TOKEN'] = 'Your Supervisely API Token'
                api = sly.Api.from_env()

                img_infos = api.pointcloud.get_list_related_images(19373403)
                ids = [img_info.id for img_info in img_infos]
                save_paths = [os.path.join("/path/to/save/", img_info.name) for img_info in img_infos]

                semaphore = asyncio.Semaphore(100)
                loop = sly.utils.get_or_create_event_loop()
                loop.run_until_complete(
                        api.pointcloud.download_related_images_async(ids, save_paths, semaphore)
                    )
        """

        if len(ids) == 0:
            return
        if len(ids) != len(paths):
            raise ValueError('Can not match "ids" and "paths" lists, len(ids) != len(paths)')
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        tasks = []
        for img_id, img_path in zip(ids, paths):
            task = self.download_related_image_async(
                img_id,
                img_path,
                semaphore,
                headers=headers,
                check_hash=check_hash,
                progress_cb=progress_cb,
                progress_cb_type=progress_cb_type,
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    def rename(
        self,
        id: int,
        name: str,
    ) -> PointcloudInfo:
        """Renames Pointcloud with given ID to a new name.

        :param id: Pointcloud ID in Supervisely.
        :type id: int
        :param name: New Pointcloud name.
        :type name: str
        :return: Information about updated Pointcloud.
        :rtype: :class:`PointcloudInfo`

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()
            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'

            pointcloud_id = 123456
            new_pointcloud_name = "3333_new.pcd"

            api.pointcloud.rename(id=pointcloud_id, name=new_pointcloud_name)
        """

        data = {
            ApiField.ID: id,
            ApiField.NAME: name,
        }

        response = self._api.post("images.editInfo", data)
        return self._convert_json_info(response.json())
