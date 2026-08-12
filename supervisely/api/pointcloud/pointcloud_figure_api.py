# coding: utf-8

"""Work with point cloud figures via the Supervisely API."""

from typing import Callable, Dict, List, Optional, Union

from requests_toolbelt import MultipartEncoder
from tqdm import tqdm

from supervisely._utils import batched
from supervisely.api.entity_annotation.figure_api import FigureApi, FigureInfo
from supervisely.api.module_api import ApiField
from supervisely.geometry.constants import INDICES
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.io.fs import decode_uint32_le, encode_uint32_le
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.video_annotation.key_id_map import KeyIdMap


class PointcloudFigureApi(FigureApi):
    """
    API for working with :class:`~supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure`.
    :class:`~supervisely.api.pointcloud.pointcloud_figure_api.PointcloudFigureApi` object is immutable.
    """

    def create(
        self,
        pointcloud_id: int,
        object_id: int,
        geometry_json: Dict,
        geometry_type: str,
        track_id: Optional[int] = None,
    ) -> int:
        """
        Create new PointcloudFigure of given point cloud object in point cloud with given ID.

        :param pointcloud_id: Point cloud ID in Supervisely.
        :type pointcloud_id: int
        :param object_id: ID of the object to which the PointcloudFigure belongs.
        :type object_id: int
        :param geometry_json: Parameters of geometry for :class:`~supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure`.
        :type geometry_json: dict
        :param geometry_type: Type of PointcloudFigure geometry.
        :type geometry_type: str
        :param track_id: int, optional.
        :type track_id: int, optional
        :returns: New figure ID
        :rtype: int

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                pcd_id = 19618685
                object_id = 5565921
                geometry_json = {'points': {'exterior': [[500, 500], [1555, 1500]], 'interior': []}}
                geometry_type = 'rectangle'

                figure_id = api.pointcloud.figure.create(pcd_id, object_id, geometry_json, geometry_type) # 643182610
        """

        return super().create(pointcloud_id, object_id, {}, geometry_json, geometry_type, track_id)

    def append_bulk(
        self,
        pointcloud_id: int,
        figures: List[PointcloudFigure],
        key_id_map: KeyIdMap,
    ) -> None:
        """
        Add PointcloudFigures to given point cloud by ID.

        :param pointcloud_id: Point cloud ID in Supervisely.
        :type pointcloud_id: int
        :param figures: List of point cloud figures to append.
        :type figures: List[:class:`~supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure`]
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`~supervisely.video_annotation.key_id_map.KeyIdMap`
        :returns: None
        :rtype: None

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                project_id = 124976
                meta_json = api.project.get_meta(project_id)
                meta = sly.ProjectMeta.from_json(meta_json)
                key_id_map = KeyIdMap()

                pcd_id = 198703212
                ann_info = api.pointcloud.annotation.download(pcd_id)
                ann = sly.PointcloudAnnotation.from_json(ann_info, meta, key_id_map)
                figures = ann.figures[:5]
                api.video.figure.append_bulk(pcd_id, figures, key_id_map)
        """

        regular_keys = []
        regular_figures_json = []
        pc_keys = []
        pc_indices = []
        pc_figures_json = []
        for figure in figures:
            figure_json = figure.to_json(key_id_map)
            if figure.geometry.name() == Pointcloud.geometry_name():
                pc_keys.append(figure.key())
                pc_indices.append(figure.geometry.indices)
                figure_json.pop(ApiField.GEOMETRY, None)
                pc_figures_json.append(figure_json)
            else:
                regular_keys.append(figure.key())
                regular_figures_json.append(figure_json)

        self._append_bulk(pointcloud_id, regular_figures_json, regular_keys, key_id_map)
        self._append_bulk(pointcloud_id, pc_figures_json, pc_keys, key_id_map)

        figure_ids = [key_id_map.get_figure_id(key) for key in pc_keys]
        if len(figure_ids) != 0:
            self.upload_indices_batch(figure_ids, pc_indices)

    def append_to_dataset(
        self,
        dataset_id: int,
        figures: List[PointcloudFigure],
        entity_ids: List[int],
        key_id_map: KeyIdMap,
    ) -> None:
        """
        Add pointcloud figures to Dataset annotations.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param figures: List of point cloud figures.
        :type figures: List[:class:`~supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure`]
        :param entity_ids: List of point cloud IDs.
        :type entity_ids: List[int]
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`~supervisely.video_annotation.key_id_map.KeyIdMap`, optional
        :rtype: None

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly
                from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
                from supervisely.pointcloud_annotation.pointcloud_annotation import PointcloudObjectCollection
                from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
                from supervisely.video_annotation.key_id_map import KeyIdMap

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()


                project_id = 17231
                dataset_id = 55875
                pointcloud_id = 19373403
                project = api.project.get_info_by_id(project_id)
                dataset = api.dataset.get_info_by_id(dataset_id)

                class_car = sly.ObjClass('car', Cuboid3d)
                classes = sly.ObjClassCollection([class_car])
                project_meta = sly.ProjectMeta(classes)
                updated_meta = api.project.update_meta(project.id, project_meta.to_json())

                key_id_map = KeyIdMap()

                car_object = sly.PointcloudObject(class_car)
                objects_collection = PointcloudObjectCollection([car_object])

                uploaded_objects_ids = api.pointcloud_episode.object.append_to_dataset(
                    dataset.id,
                    objects_collection,
                    key_id_map,
                )

                position, rotation, dimension = Vector3d(-32.4, 33.9, -0.7), Vector3d(0., 0, 0.1), Vector3d(1.8, 3.9, 1.6)
                cuboid = Cuboid3d(position, rotation, dimension)
                figure_1 = PointcloudFigure(car_object, cuboid)

                api.pointcloud_episode.figure.append_to_dataset(
                    dataset.id,
                    [figure_1],
                    [pointcloud_id],
                    key_id_map,
                )
        """

        regular_keys = []
        regular_figures_json = []
        pc_keys = []
        pc_indices = []
        pc_figures_json = []
        for figure, entity_id in zip(figures, entity_ids):
            figure_json = figure.to_json(key_id_map)
            figure_json[ApiField.ENTITY_ID] = entity_id
            if figure_json.get(ApiField.GEOMETRY_TYPE) == Pointcloud.geometry_name():
                pc_keys.append(figure.key())
                pc_indices.append(figure.geometry.indices)
                figure_json.pop(ApiField.GEOMETRY, None)
                pc_figures_json.append(figure_json)
            else:
                regular_keys.append(figure.key())
                regular_figures_json.append(figure_json)

        self._append_bulk(
            dataset_id,
            regular_figures_json,
            regular_keys,
            key_id_map,
            field_name=ApiField.DATASET_ID,
        )
        self._append_bulk(
            dataset_id, pc_figures_json, pc_keys, key_id_map, field_name=ApiField.DATASET_ID
        )

        figure_ids = [key_id_map.get_figure_id(key) for key in pc_keys]
        if len(figure_ids) != 0:
            self.upload_indices_batch(figure_ids, pc_indices)

    def _convert_json_info(self, info: dict, skip_missing=True):
        return super()._convert_json_info(info, skip_missing)

    def download(
        self,
        dataset_id: int,
        pointcloud_ids: List[int] = None,
        skip_geometry: bool = False,
        **kwargs
    ) -> Dict[int, List[FigureInfo]]:
        """
        Method returns a dictionary with pairs of pointcloud ID and list of FigureInfo for the given dataset ID. Can be filtered by pointcloud IDs.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param pointcloud_ids: Specify the list of pointcloud IDs within the given dataset ID. If pointcloud_ids is None, the method returns all possible pairs of images with figures. Note: Consider using `sly.batched()` to ensure that no figures are lost in the response.
        :type pointcloud_ids: List[int], optional
        :param skip_geometry: Skip the download of figure geometry. May be useful for a significant api request speed increase in the large datasets.
        :type skip_geometry: bool

        :returns: A dictionary where keys are pointcloud IDs and values are lists of figures.
        :rtype: Dict[int, List[:class:`~supervisely.api.entity_annotation.figure_api.FigureInfo`]]
        """
        if kwargs.get("image_ids", False) is not False:
            pointcloud_ids = kwargs["image_ids"]  # backward compatibility
        figures = super().download(dataset_id, pointcloud_ids, skip_geometry)
        if skip_geometry:
            return figures
        return self.hydrate_figure_infos_dict(figures)

    def upload_indices_batch(self, figure_ids: List[int], indices_batch: List[List[int]]) -> None:
        """
        Upload point cloud figure geometry as raw little-endian uint32 index data to storage.

        :param figure_ids: List of figure IDs in Supervisely.
        :type figure_ids: List[int]
        :param indices_batch: Point indices per figure, aligned with ``figure_ids``.
        :type indices_batch: List[List[int]]
        :returns: None
        :rtype: None
        """
        if len(figure_ids) != len(indices_batch):
            raise ValueError(
                f"figure_ids and indices_batch must have the same length: "
                f"{len(figure_ids)} != {len(indices_batch)}."
            )

        for batch in batched(list(zip(figure_ids, indices_batch)), batch_size=50):
            fields = []
            for figure_id, indices in batch:
                fields.append((ApiField.FIGURE_ID, str(figure_id)))
                fields.append(
                    (
                        ApiField.GEOMETRY,
                        (str(figure_id), encode_uint32_le(indices), "application/octet-stream"),
                    )
                )
            encoder = MultipartEncoder(fields=fields)
            self._api.post("figures.bulk.upload.geometry", encoder)

    def download_indices_batch(
        self,
        figure_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[List[int]]:
        """
        Download point cloud figure geometry stored as raw little-endian uint32 index data.

        Progress is updated by one for each downloaded figure geometry.

        :param figure_ids: List of figure IDs in Supervisely.
        :type figure_ids: List[int]
        :param progress_cb: Progress bar or callback to track download progress.
        :type progress_cb: Union[tqdm, Callable], optional
        :returns: Point indices per figure, aligned with ``figure_ids``.
        :rtype: List[List[int]]
        """
        geometries = {}
        for figure_id, part in self._download_geometries_generator(figure_ids):
            geometries[figure_id] = decode_uint32_le(part.content)
            if progress_cb is not None:
                if hasattr(progress_cb, "update") and callable(getattr(progress_cb, "update")):
                    progress_cb.update(1)
                elif callable(progress_cb):
                    progress_cb(1)

        if len(geometries) != len(figure_ids):
            raise RuntimeError("Not all point cloud geometries were downloaded")
        return [geometries[figure_id] for figure_id in figure_ids]

    def hydrate_figure_infos_dict(
        self, figures_by_entity: Dict[int, List[FigureInfo]]
    ) -> Dict[int, List[FigureInfo]]:
        """
        Hydrate separately stored point-cloud indices in downloaded figure mappings.
        """
        refs = []
        for entity_id, figures in figures_by_entity.items():
            for idx, figure in enumerate(figures):
                if self._should_hydrate_figure_info(figure):
                    refs.append((entity_id, idx, figure.id))

        if len(refs) == 0:
            return figures_by_entity

        hydrated = {entity_id: list(figures) for entity_id, figures in figures_by_entity.items()}
        indices_batch = self.download_indices_batch([figure_id for _, _, figure_id in refs])
        for (entity_id, idx, _), indices in zip(refs, indices_batch):
            hydrated[entity_id][idx] = hydrated[entity_id][idx]._replace(
                geometry={INDICES: indices}
            )
        return hydrated

    def inject_geometries_into_annotations(self, anns_json: List[Dict]) -> List[Dict]:
        """
        Patch ``point_cloud`` figures whose index geometry is stored separately (not inline)
        by downloading the indices and writing them back into the annotation JSON in-place.

        Figures that already carry inline indices (old format) are left untouched, so both
        the old and the new server formats are supported.

        :param anns_json: List of annotation JSONs (each with ``frames`` -> ``figures``).
        :type anns_json: List[dict]
        :returns: The same list with separately stored geometries injected.
        :rtype: List[dict]
        """
        refs = []
        for ann in anns_json:
            if not isinstance(ann, dict):
                continue
            for figure_json in self._iter_pointcloud_figures(ann):
                if figure_json.get(ApiField.GEOMETRY_TYPE) != Pointcloud.geometry_name():
                    continue
                if self._extract_indices(figure_json.get(ApiField.GEOMETRY)) is not None:
                    continue
                figure_id = figure_json.get(ApiField.ID)
                if figure_id is None:
                    continue
                refs.append((figure_id, figure_json))

        if len(refs) == 0:
            return anns_json

        figure_ids = [figure_id for figure_id, _ in refs]
        indices_batch = self.download_indices_batch(figure_ids)
        for (_, figure_json), indices in zip(refs, indices_batch):
            figure_json[ApiField.GEOMETRY] = {INDICES: indices}
        return anns_json

    @staticmethod
    def _iter_pointcloud_figures(ann: Dict):
        """
        Yield figure-level JSON dicts from a point cloud annotation, supporting both
        shapes:
        - Single point cloud annotation: figures at top level (``ann["figures"]``).
        - Episode annotation: figures inside frames (``ann["frames"][i]["figures"]``).
        """
        frames = ann.get(ApiField.FRAMES)
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                for figure_json in frame.get(ApiField.FIGURES, []) or []:
                    if isinstance(figure_json, dict):
                        yield figure_json
            return
        for figure_json in ann.get(ApiField.FIGURES, []) or []:
            if isinstance(figure_json, dict):
                yield figure_json

    @staticmethod
    def _extract_indices(geometry) -> Optional[List[int]]:
        if not isinstance(geometry, dict):
            return None
        indices = geometry.get(INDICES)
        if isinstance(indices, list):
            return indices
        return None

    @staticmethod
    def _should_hydrate_figure_info(figure: FigureInfo) -> bool:
        if figure.geometry_type != Pointcloud.geometry_name() or figure.id is None:
            return False
        return PointcloudFigureApi._extract_indices(figure.geometry) is None
