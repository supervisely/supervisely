from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from supervisely import (
    ObjClass,
    PointcloudEpisodeAnnotation,
    PointcloudEpisodeFrame,
    PointcloudEpisodeFrameCollection,
    PointcloudEpisodeObject,
    PointcloudEpisodeObjectCollection,
    PointcloudFigure,
    ProjectMeta,
    is_development,
    logger,
)
from supervisely.api.api import Api
from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud.kitti_3d.kitti_3d_helper import convert_bin_to_pcd
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.convert.pointcloud_episodes.semantic_kitti.semantic_kitti_helper import (
    generate_color_for_class,
    read_label_file,
)
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.io.fs import (
    file_exists,
    get_file_ext,
    list_files_recursively,
    silent_remove,
)
from supervisely.project.project_settings import LabelingInterface


class SemanticKITTIConverter(PointcloudEpisodeConverter):
    """Converter for SemanticKITTI pointcloud episodes with semantic segmentation."""

    class Item:
        """Parsed SemanticKITTI sequence bundle."""

        def __init__(
            self,
            sequence_name: str,
            frame_paths: List[str],
            label_paths: List[Optional[str]],
            poses_path: Optional[str] = None,
            times_path: Optional[str] = None,
            custom_data: Optional[dict] = None,
        ):
            """:param sequence_name: Sequence identifier.
            :type sequence_name: str
            :param frame_paths: Paths to ``.bin`` pointcloud frames.
            :type frame_paths: List[str]
            :param label_paths: Paths to ``.label`` annotation files.
            :type label_paths: List[Optional[str]]
            :param poses_path: Path to poses file.
            :type poses_path: str, optional
            :param times_path: Path to timestamps file.
            :type times_path: str, optional
            :param custom_data: Extra data.
            :type custom_data: dict, optional
            """
            self._sequence_name = sequence_name
            self._frame_paths = frame_paths
            self._label_paths = label_paths
            self._poses_path = poses_path
            self._times_path = times_path
            self._type = "point_cloud_episode"
            self._custom_data = custom_data if custom_data is not None else {}

        @property
        def sequence_name(self) -> str:
            return self._sequence_name

        @property
        def frame_paths(self) -> List[str]:
            return self._frame_paths

        @property
        def label_paths(self) -> List[Optional[str]]:
            return self._label_paths

        @property
        def frame_count(self) -> int:
            return len(self._frame_paths)

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]],
        upload_as_links: bool = False,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        """See :class:`~supervisely.convert.base_converter.BaseConverter` for params."""
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)
        self._label_map = {}

    def __str__(self) -> str:
        return AvailablePointcloudEpisodesConverters.SEMANTIC_KITTI

    @property
    def key_file_ext(self) -> str:
        return ".bin"

    def validate_format(self) -> bool:
        self._items = []
        filter_fn = (
            lambda f: get_file_ext(f) == self.key_file_ext and Path(f).parent.name == "velodyne"
        )
        velodyne_files = list_files_recursively(self._input_data, filter_fn=filter_fn)
        sequences = sorted(set(Path(f).parent.parent for f in velodyne_files))
        if not sequences:
            return False

        all_semantic_ids = set()

        for sequence_path in sequences:
            sequence_name = sequence_path.name
            velodyne_dir = sequence_path / "velodyne"
            labels_dir = sequence_path / "labels"
            poses_path = sequence_path / "poses.txt"
            times_path = sequence_path / "times.txt"

            bin_files = sorted(velodyne_dir.glob("*.bin"))
            frame_paths = [str(f) for f in bin_files]

            label_paths = []
            for bin_file in bin_files:
                label_file = labels_dir / f"{bin_file.stem}.label"
                label_paths.append(str(label_file) if label_file.exists() else None)

            if labels_dir.exists():
                for label_path in label_paths:
                    if label_path is not None:
                        semantic_labels, _ = read_label_file(label_path)
                        if semantic_labels is not None:
                            all_semantic_ids.update(np.unique(semantic_labels))

            item = self.Item(
                sequence_name=sequence_name,
                frame_paths=frame_paths,
                label_paths=label_paths,
                poses_path=str(poses_path) if poses_path.exists() else None,
                times_path=str(times_path) if times_path.exists() else None,
            )
            self._items.append(item)

        if not self._items:
            return False

        label_map = {}
        for sem_id in sorted(all_semantic_ids):
            sem_id = int(sem_id)
            label_map[sem_id] = (f"class_{sem_id}", generate_color_for_class(sem_id))

        self._label_map = label_map

        obj_classes = []
        for class_id in sorted(self._label_map.keys()):
            class_name, class_color = self._label_map[class_id]
            obj_class = ObjClass(class_name, geometry_type=Pointcloud, color=class_color)
            obj_classes.append(obj_class)

        self._meta = ProjectMeta(obj_classes=obj_classes)
        return len(self._items) > 0

    def to_supervisely(
        self,
        item: Item,
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
    ) -> PointcloudEpisodeAnnotation:
        """Convert to Supervisely format."""
        class_id_to_obj: Dict[int, PointcloudEpisodeObject] = {}
        frames = []

        for idx, label_path in enumerate(item.label_paths):
            figures = []

            if label_path is not None:
                semantic_labels, _ = read_label_file(label_path)
                if semantic_labels is not None:
                    for class_id in np.unique(semantic_labels):
                        class_id = int(class_id)
                        if class_id not in self._label_map:
                            continue
                        if class_id not in class_id_to_obj:
                            class_name, _ = self._label_map[class_id]
                            class_name = renamed_classes.get(class_name, class_name)
                            obj_class = meta.get_obj_class(class_name)
                            if obj_class is None:
                                continue
                            class_id_to_obj[class_id] = PointcloudEpisodeObject(obj_class)

                        pcd_obj = class_id_to_obj[class_id]
                        point_indices = np.where(semantic_labels == class_id)[0]
                        if len(point_indices) > 0:
                            geom = Pointcloud(indices=point_indices.tolist())
                            figure = PointcloudFigure(pcd_obj, geom, frame_index=idx)
                            figures.append(figure)

            frames.append(PointcloudEpisodeFrame(idx, figures=figures))

        return PointcloudEpisodeAnnotation(
            frames_count=item.frame_count,
            objects=PointcloudEpisodeObjectCollection(
                [class_id_to_obj[class_id] for class_id in sorted(class_id_to_obj)]
            ),
            frames=PointcloudEpisodeFrameCollection(frames),
        )

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 1,
        log_progress=True,
    ):
        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        dataset_info = api.dataset.get_info_by_id(dataset_id)

        if log_progress:
            total_frames = sum([item.frame_count for item in self._items])
            progress, progress_cb = self.get_progress(
                total_frames, "Converting SemanticKITTI episodes..."
            )
        else:
            progress_cb = None

        multiple_sequences = len(self._items) > 1

        for item in self._items:
            if multiple_sequences:
                sequence_ds = api.dataset.create(
                    dataset_info.project_id, item.sequence_name, parent_id=dataset_id
                )
            else:
                sequence_ds = dataset_info

            frame_to_pcd_ids = {}

            for idx, bin_path in enumerate(item.frame_paths):
                pcd_path = str(Path(bin_path).with_suffix(".pcd"))
                if file_exists(pcd_path):
                    logger.warning(f"Overwriting file: {pcd_path}")

                convert_bin_to_pcd(bin_path, pcd_path)

                pcd_name = Path(pcd_path).name
                info = api.pointcloud_episode.upload_path(
                    sequence_ds.id, pcd_name, pcd_path, {"frame": idx}
                )
                frame_to_pcd_ids[idx] = info.id

                silent_remove(pcd_path)

                if log_progress:
                    progress_cb(1)

            try:
                ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                api.pointcloud_episode.annotation.append(sequence_ds.id, ann, frame_to_pcd_ids)
            except Exception as e:
                logger.error(
                    f"Failed to upload annotation for sequence: {sequence_ds.name}. Error: {repr(e)}",
                    stack_info=False,
                )
                continue

            logger.info(f"Dataset ID:{sequence_ds.id} has been successfully uploaded.")

        if log_progress:
            if is_development():
                progress.close()
