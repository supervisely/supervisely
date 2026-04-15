"""
SemanticKITTI format converter for Supervisely Point Cloud Episodes.

Converts SemanticKITTI-format point cloud sequences to Supervisely Episodes with:
- Semantic segmentation via RGB point coloring
- Instance tracking across frames
- Auto-discovery of classes from data
- Auto-generation of colors for unknown classes

Follows Supervisely SDK converter architecture (similar to KITTI-360).
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

from supervisely import (
    ObjClass,
    ObjClassCollection,
    ProjectMeta,
    logger,
    is_development,
)
from supervisely.api.api import Api
from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.project.project_settings import LabelingInterface
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.io.fs import (
    file_exists,
    silent_remove,
)
from supervisely.pointcloud_annotation.pointcloud_episode_annotation import (
    PointcloudEpisodeAnnotation,
)
from supervisely.pointcloud_annotation.pointcloud_episode_frame import (
    PointcloudEpisodeFrame,
)
from supervisely.pointcloud_annotation.pointcloud_episode_frame_collection import (
    PointcloudEpisodeFrameCollection,
)
from supervisely.pointcloud_annotation.pointcloud_episode_object import (
    PointcloudEpisodeObject,
)
from supervisely.pointcloud_annotation.pointcloud_episode_object_collection import (
    PointcloudEpisodeObjectCollection,
)
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure

# Import helper functions
from supervisely.convert.pointcloud_episodes.semantic_kitti.semantic_kitti_helper import (
    read_bin_pointcloud,
    read_label_file,
    convert_bin_to_pcd,
    scan_labels_for_classes,
    validate_sequence_structure,
)

# Default label mapping can be customized or auto-generated
DEFAULT_SEMANTIC_KITTI_LABELS = {}


class SemanticKITTIConverter(PointcloudEpisodeConverter):
    """
    Converter for SemanticKITTI point cloud episodes.

    Converts sequences of point clouds with semantic/instance labels to Supervisely format.
    Points are colored by semantic class. No 3D bounding boxes, only semantic segmentation.

    Data format:
        sequence/
        ├── velodyne/      # .bin files: N×4 float32 (x, y, z, intensity)
        ├── labels/        # .label files: N×1 uint32 (semantic + instance IDs)
        ├── poses.txt      # Camera poses (12 values per line, one line per frame)
        └── times.txt      # Timestamps (one per line)

    Features:
        - Auto-discovers classes from .label files
        - Auto-generates colors for unknown classes using HSV golden angle
        - Semantic segmentation via point RGB colors
        - Works with any SemanticKITTI-format data (not just original dataset)

    Example:
        >>> converter = SemanticKITTIConverter(
        ...     "path/to/sequences/00",
        ...     labeling_interface=None
        ... )
        >>> converter.validate_format()
        True
        >>> api = Api.from_env()
        >>> converter.upload_dataset(api, dataset_id)
    """

    class Item:
        """
        Represents a SemanticKITTI sequence with all frames and metadata.

        Stores paths to all point clouds in the sequence plus associated labels.
        """

        def __init__(
            self,
            sequence_name: str,
            frame_paths: List[str],
            label_paths: List[str],
            poses_path: Optional[str] = None,
            times_path: Optional[str] = None,
            custom_data: Optional[dict] = None,
        ):
            """
            Args:
                sequence_name: Sequence identifier (e.g., "00", "01")
                frame_paths: List of paths to .bin point cloud files
                label_paths: List of paths to .label annotation files
                poses_path: Optional path to poses.txt file (camera poses)
                times_path: Optional path to times.txt file (timestamps)
                custom_data: Optional extra data
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
        def label_paths(self) -> List[str]:
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
        label_map: Optional[Dict] = None,
    ):
        """
        Initialize SemanticKITTI converter.

        Args:
            input_data: Path to sequence directory or directory containing sequences
            labeling_interface: Supervisely labeling interface (optional)
            upload_as_links: Whether to upload as links (not supported for this format)
            remote_files_map: Remote files mapping (optional)
            label_map: Optional custom class mapping {class_id: (name, [r,g,b])}
                      None = use auto-generation for all classes
                      dict = use your definitions

        Example:
            >>> # Use auto-generated classes
            >>> converter = SemanticKITTIConverter("sequences/00", None)

            >>> # Custom classes
            >>> converter = SemanticKITTIConverter("sequences/00", None, label_map={
            ...     1: ("robot", [255, 0, 0]),
            ...     2: ("obstacle", [0, 255, 0])
            ... })
        """
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        # Label mapping: None = use defaults, {} = auto-gen all, dict = custom
        if label_map is None:
            self._label_map = DEFAULT_SEMANTIC_KITTI_LABELS.copy()
        else:
            self._label_map = label_map.copy()

        self._items = []
        self._meta = None

    def __str__(self) -> str:
        return AvailablePointcloudEpisodesConverters.SEMANTIC_KITTI

    @property
    def key_file_ext(self) -> str:
        """File extension that identifies this format."""
        return ".bin"

    def validate_format(self) -> bool:
        """
        Validate input data format and parse sequence(s).

        Checks for required directory structure and files.
        Scans label files to discover classes and build ProjectMeta.

        Returns:
            bool: True if format is valid and data can be converted
        """
        self._items = []
        input_path = Path(self._input_data)

        # Check if input is a single sequence or directory with sequences
        if (input_path / "velodyne").exists():
            # Single sequence directory
            sequences = [input_path]
        else:
            # Directory containing multiple sequences
            # Look for subdirectories with velodyne/ inside
            sequences = [
                d for d in input_path.iterdir() if d.is_dir() and (d / "velodyne").exists()
            ]

        if len(sequences) == 0:
            logger.error(f"No valid sequences found in {input_path}")
            return False

        logger.info(f"Found {len(sequences)} sequence(s)")

        # Collect all unique semantic IDs from all sequences
        all_semantic_ids = set()

        # Process each sequence
        for sequence_path in sequences:
            sequence_name = sequence_path.name

            # Validate structure
            is_valid, error_msg = validate_sequence_structure(sequence_path)
            if not is_valid:
                logger.error(f"Sequence {sequence_name} validation failed: {error_msg}")
                continue

            # Get paths
            velodyne_dir = sequence_path / "velodyne"
            labels_dir = sequence_path / "labels"
            poses_path = sequence_path / "poses.txt"
            times_path = sequence_path / "times.txt"

            # Get frame files
            bin_files = sorted(velodyne_dir.glob("*.bin"))
            frame_paths = [str(f) for f in bin_files]

            # Get corresponding label files
            label_paths = []
            for bin_file in bin_files:
                label_file = labels_dir / f"{bin_file.stem}.label"
                label_paths.append(str(label_file) if label_file.exists() else None)

            # Scan labels to find classes
            if labels_dir.exists():
                # Read all label files to find unique semantic IDs
                for label_path in label_paths:
                    if label_path is not None:
                        semantic_labels, _ = read_label_file(label_path)
                        if semantic_labels is not None:
                            all_semantic_ids.update(np.unique(semantic_labels))

            # Create Item for this sequence
            item = self.Item(
                sequence_name=sequence_name,
                frame_paths=frame_paths,
                label_paths=label_paths,
                poses_path=str(poses_path) if poses_path.exists() else None,
                times_path=str(times_path) if times_path.exists() else None,
            )
            self._items.append(item)

            logger.info(f"  - {sequence_name}: {len(frame_paths)} frames")

        if len(self._items) == 0:
            logger.error("No valid sequences found")
            return False

        # Build complete label mapping from discovered classes
        complete_label_map = self._label_map.copy()
        auto_generated = 0

        for sem_id in sorted(all_semantic_ids):
            sem_id = int(sem_id)
            if sem_id not in complete_label_map:
                # Auto-generate class
                from supervisely.convert.pointcloud_episodes.semantic_kitti.semantic_kitti_helper import (
                    generate_color_for_class,
                )

                complete_label_map[sem_id] = (
                    f"class_{sem_id}",
                    generate_color_for_class(sem_id),
                )
                auto_generated += 1

        # Update label map with complete mapping
        self._label_map = complete_label_map

        logger.info(f"Total classes: {len(all_semantic_ids)}")
        logger.info(f"  - Predefined: {len(all_semantic_ids) - auto_generated}")
        logger.info(f"  - Auto-generated: {auto_generated}")

        # Create ProjectMeta with all classes — geometry is Pointcloud (point indices mask)
        obj_classes = []
        for class_id in sorted(self._label_map.keys()):
            class_name, class_color = self._label_map[class_id]
            obj_class = ObjClass(class_name, geometry_type=Pointcloud, color=class_color)
            obj_classes.append(obj_class)

        self._meta = ProjectMeta(obj_classes=ObjClassCollection(obj_classes))

        return len(self._items) > 0

    def to_supervisely(
        self,
        item: Item,
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
    ) -> PointcloudEpisodeAnnotation:
        """
        Convert SemanticKITTI sequence to Supervisely annotation format.

        Creates one PointcloudEpisodeObject per semantic class, tracked across all frames.
        For each frame, creates PointcloudFigure objects with Pointcloud(indices) geometry —
        a list of point indices belonging to that class in this frame.
        These appear in the 3D editor as editable semantic segmentation masks.

        Args:
            item: Sequence item to convert
            meta: Project metadata
            renamed_classes: Class name remapping (unused)
            renamed_tags: Tag name remapping (unused)

        Returns:
            PointcloudEpisodeAnnotation with per-frame segmentation figures
        """
        import numpy as np

        # Create one tracked object per semantic class
        class_id_to_obj: Dict[int, PointcloudEpisodeObject] = {}
        for class_id, (class_name, _) in self._label_map.items():
            obj_class = meta.get_obj_class(class_name)
            if obj_class is not None:
                class_id_to_obj[class_id] = PointcloudEpisodeObject(obj_class)

        frames = []
        for idx, label_path in enumerate(item.label_paths):
            figures = []

            if label_path is not None and Path(label_path).exists():
                semantic_labels, _ = read_label_file(label_path)
                if semantic_labels is not None:
                    # For each class present in this frame, collect point indices
                    for class_id, pcd_obj in class_id_to_obj.items():
                        point_indices = np.where(semantic_labels == class_id)[0]
                        if len(point_indices) > 0:
                            geom = Pointcloud(indices=point_indices.tolist())
                            figure = PointcloudFigure(pcd_obj, geom, frame_index=idx)
                            figures.append(figure)

            frames.append(PointcloudEpisodeFrame(idx, figures=figures))

        return PointcloudEpisodeAnnotation(
            frames_count=item.frame_count,
            objects=PointcloudEpisodeObjectCollection(list(class_id_to_obj.values())),
            frames=PointcloudEpisodeFrameCollection(frames),
        )

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 1,
        log_progress=True,
    ):
        """
        Upload converted SemanticKITTI sequence(s) to Supervisely platform.

        - Converts .bin files to raw .pcd (XYZ + intensity, no color encoding)
        - Uploads point clouds as a Point Cloud Episode
        - Creates Supervisely annotations: per-frame figures with point indices per class
        - Annotations are fully editable in the Supervisely 3D labeling tool
        - Creates nested datasets if multiple sequences

        Args:
            api: Supervisely API instance
            dataset_id: Target dataset ID
            batch_size: Batch size for uploading (default: 1)
            log_progress: Whether to show progress bar
        """
        import numpy as np

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
            # Create nested dataset for each sequence if multiple
            if multiple_sequences:
                sequence_ds = api.dataset.create(
                    dataset_info.project_id, item.sequence_name, parent_id=dataset_id
                )
            else:
                sequence_ds = dataset_info

            frame_to_pcd_ids = {}

            # Process each frame
            for idx, (bin_path, label_path) in enumerate(zip(item.frame_paths, item.label_paths)):
                # Convert .bin to .pcd format
                pcd_path = str(Path(bin_path).with_suffix(".pcd"))
                if file_exists(pcd_path):
                    logger.warning(f"Overwriting file: {pcd_path}")

                convert_bin_to_pcd(bin_path, pcd_path)

                # Upload point cloud
                pcd_name = Path(pcd_path).name
                info = api.pointcloud_episode.upload_path(
                    sequence_ds.id, pcd_name, pcd_path, {"frame": idx}
                )
                frame_to_pcd_ids[idx] = info.id

                # Clean up temporary PCD file
                silent_remove(pcd_path)

                if log_progress:
                    progress_cb(1)

            # Create and upload episode annotation
            try:
                ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                api.pointcloud_episode.annotation.append(sequence_ds.id, ann, frame_to_pcd_ids)
            except Exception as e:
                logger.error(
                    f"Failed to upload annotation for sequence: {sequence_ds.name}. Error: {repr(e)}",
                    stack_info=False,
                )
                continue

            logger.info(
                f"Dataset '{sequence_ds.name}' (ID: {sequence_ds.id}) uploaded successfully"
            )

        if log_progress:
            if is_development():
                progress.close()

        logger.info("✓ All sequences uploaded!")
        logger.info("Semantic segmentation masks are saved as editable annotations.")
        logger.info("Open the project in Supervisely 3D viewer to review and edit labels.")


# Convenience function for quick conversion
def import_semantickitti(
    api: Api,
    workspace_id: int,
    input_path: str,
    project_name: str = "SemanticKITTI",
    label_map: Optional[Dict] = None,
) -> tuple:
    """
    Import SemanticKITTI sequence(s) to Supervisely in one call.

    Args:
        api: Supervisely API instance
        workspace_id: Target workspace ID
        input_path: Path to sequence or directory with sequences
        project_name: Name for created project
        label_map: Optional custom class mapping

    Returns:
        Tuple of (project_info, dataset_info)

    Example:
        >>> from supervisely import Api
        >>> api = Api.from_env()
        >>> project, dataset = import_semantickitti(
        ...     api,
        ...     workspace_id=12345,
        ...     input_path="data/sequences/00",
        ...     project_name="My Sequence"
        ... )
    """
    # Create converter
    converter = SemanticKITTIConverter(input_path, label_map=label_map)

    # Validate format
    if not converter.validate_format():
        raise ValueError(f"Invalid SemanticKITTI format: {input_path}")

    # Create project
    project_info = api.project.create(
        workspace_id,
        project_name,
        type="point_cloud_episodes",
        change_name_if_conflict=True,
    )

    # Upload metadata
    api.project.update_meta(project_info.id, converter._meta)

    # Create dataset
    dataset_info = api.dataset.create(project_info.id, "episodes")

    # Upload data
    converter.upload_dataset(api, dataset_info.id)

    return project_info, dataset_info
