from __future__ import annotations
import os
import time
import cv2
from typing import Dict, List, Tuple, Union, Optional, TypeVar, Any

from supervisely.api.api import Api
from supervisely.annotation.annotation import Annotation
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_tag_collection import VideoTagCollection
from supervisely.video_annotation.video_tag import VideoTag
from supervisely.video_annotation.video_annotation import VideoAnnotation
from supervisely.video_annotation.video_object import VideoObject
from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.video_figure import VideoFigure
from supervisely.video_annotation.video_tag import VideoTag

from supervisely.io.fs import ensure_base_path
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.io.fs import list_files_recursively, mkdir, dir_exists, get_file_name_with_ext, copy_file
from supervisely.imaging import image as sly_image
from supervisely.video import video as sly_video
from supervisely.project.project import Project, ProjectType, Dataset, OpenMode
from supervisely.project.video_project import VideoProject, VideoDataset
from supervisely.nn.inference.inference import Inference
from supervisely import logger
import numpy as np

class LocalPredictor:
    """Class for performing local predictions using Supervisely models.
    
    This class handles various types of input data for prediction:
    - Project ID in the Supervisely platform
    - Dataset ID in the Supervisely platform
    - Image ID in the Supervisely platform
    - Local image and video files
    - Local directories with images and videos
    - Local Supervisely project
    """
    
    def __init__(self, inference_instance: Inference):
        """
        Initialize LocalPredictor.
        
        :param inference_instance: Instance of the Inference class to be used for predictions
        :type inference_instance: :class:`Inference`
        """
        self.inference = inference_instance
        self.api = inference_instance.api
        self._args = inference_instance._args

    def predict_by_args(self) -> None:
        """
        Main method to run prediction based on command line arguments.
        Analyzes arguments and calls the corresponding prediction method.
        """
        if self._args.project_id is not None:
            self.predict_project_id_by_args(
                self.api,
                self._args.project_id,
                None,
                self._args.output,
                self._args.settings,
                self._args.draw,
                self._args.upload,
            )
        elif self._args.dataset_id is not None:
            self.predict_dataset_id_by_args(
                self.api,
                self._args.dataset_id,
                self._args.output,
                self._args.settings,
                self._args.draw,
                self._args.upload,
            )
        elif self._args.image_id is not None:
            self.predict_image_id_by_args(
                self.api,
                self._args.image_id,
                self._args.output,
                self._args.settings,
                self._args.draw,
                self._args.upload,
            )
        elif self._args.input is not None:
            self.predict_local_data_by_args(
                self._args.input,
                self._args.settings,
                self._args.output,
                self._args.draw,
            )

    def predict_project_id_by_args(
        self,
        api: Api,
        project_id: int,
        dataset_ids: Optional[List[int]] = None,
        output_dir: str = "./predictions",
        settings: Optional[str] = None,
        draw: bool = False,
        upload: bool = False,
    ) -> None:
        """Predict an entire project or selected datasets in a project.
        
        :param api: Supervisely API instance
        :type api: Api
        :param project_id: Project ID in Supervisely
        :type project_id: int
        :param dataset_ids: List of dataset IDs to predict (if None, all datasets are used)
        :type dataset_ids: Optional[List[int]]
        :param output_dir: Directory for saving results
        :type output_dir: str
        :param settings: Prediction settings string or path to settings file
        :type settings: Optional[str]
        :param draw: Flag for visualization of predictions (not supported for projects)
        :type draw: bool
        :param upload: Flag for uploading prediction results to the platform
        :type upload: bool
        :raises ValueError: If API is not initialized or visualization is requested for a project
        """
        missing_env_message = "Set 'SERVER_ADDRESS' and 'API_TOKEN' environment variables to predict data on Supervisely platform."
        if self.api is None:
            raise ValueError(missing_env_message)

        if dataset_ids:
            logger.info(f"Predicting datasets: '{dataset_ids}'")
        else:
            logger.info(f"Predicting project: '{project_id}'")

        if draw:
            raise ValueError("Draw visualization is not supported for project inference")

        state = {
            "projectId": project_id,
            "dataset_ids": dataset_ids,
            "settings": settings,
        }
        if upload:
            source_project = api.project.get_info_by_id(project_id)
            workspace_id = source_project.workspace_id
            output_project = api.project.create(
                workspace_id,
                f"{source_project.name} predicted",
                change_name_if_conflict=True,
            )
            state["output_project_id"] = output_project.id
        results = self.inference._inference_project_id(api=self.api, state=state)

        dataset_infos = api.dataset.get_list(project_id)
        datasets_map = {dataset_info.id: dataset_info.name for dataset_info in dataset_infos}

        if not upload:
            for prediction in results:
                dataset_name = datasets_map[prediction["dataset_id"]]
                image_name = prediction["image_name"]
                pred_dir = os.path.join(output_dir, dataset_name)
                pred_path = os.path.join(pred_dir, f"{image_name}.json")
                ann_json = prediction["annotation"]
                if not dir_exists(pred_dir):
                    mkdir(pred_dir)
                dump_json_file(ann_json, pred_path)

    def predict_dataset_id_by_args(
        self,
        api: Api,
        dataset_ids: List[int],
        output_dir: str = "./predictions",
        settings: Optional[str] = None,
        draw: bool = False,
        upload: bool = False,
    ) -> None:
        """Predict specified datasets.
        
        :param api: Supervisely API instance
        :type api: Api
        :param dataset_ids: List of dataset IDs to predict
        :type dataset_ids: List[int]
        :param output_dir: Directory for saving results
        :type output_dir: str
        :param settings: Prediction settings string or path to settings file
        :type settings: Optional[str]
        :param draw: Flag for visualization of predictions (not supported for datasets)
        :type draw: bool
        :param upload: Flag for uploading prediction results to the platform
        :type upload: bool
        :raises ValueError: If API is not initialized, visualization is requested, or datasets are from different projects
        """
        missing_env_message = "Set 'SERVER_ADDRESS' and 'API_TOKEN' environment variables to predict data on Supervisely platform."
        if draw:
            raise ValueError("Draw visualization is not supported for dataset inference")
        if self.api is None:
            raise ValueError(missing_env_message)
        dataset_infos = [api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids]
        project_ids = list(set([dataset_info.project_id for dataset_info in dataset_infos]))
        if len(project_ids) > 1:
            raise ValueError("All datasets should belong to the same project")
        self.predict_project_id_by_args(
            api, project_ids[0], dataset_ids, output_dir, settings, draw, upload
        )

    def predict_image_id_by_args(
        self,
        api: Api,
        image_id: int,
        output_dir: str = "./predictions",
        settings: Optional[str] = None,
        draw: bool = False,
        upload: bool = False,
    ) -> None:
        """Predict a single image by its ID.
        
        :param api: Supervisely API instance
        :type api: Api
        :param image_id: Image ID in Supervisely
        :type image_id: int
        :param output_dir: Directory for saving results
        :type output_dir: str
        :param settings: Prediction settings string or path to settings file
        :type settings: Optional[str]
        :param draw: Flag for visualization of predictions
        :type draw: bool
        :param upload: Flag for uploading prediction results to the platform
        :type upload: bool
        :raises ValueError: If API is not initialized
        """
        missing_env_message = "Set 'SERVER_ADDRESS' and 'API_TOKEN' environment variables to predict data on Supervisely platform."
        if self.api is None:
            raise ValueError(missing_env_message)

        logger.info(f"Predicting image: '{image_id}'")

        image_np = api.image.download_np(image_id)
        ann = self._predict_image_np(image_np, settings)

        image_info = None
        if not upload:
            ann_json = ann.to_json()
            image_info = api.image.get_info_by_id(image_id)
            dataset_info = api.dataset.get_info_by_id(image_info.dataset_id)
            pred_dir = os.path.join(output_dir, dataset_info.name)
            pred_path = os.path.join(pred_dir, f"{image_info.name}.json")
            if not dir_exists(pred_dir):
                mkdir(pred_dir)
            dump_json_file(ann_json, pred_path)

        if draw:
            if image_info is None:
                image_info = api.image.get_info_by_id(image_id)
                dataset_info = api.dataset.get_info_by_id(image_info.dataset_id)
            vis_path = os.path.join(output_dir, dataset_info.name, f"{image_info.name}.png")
            ann.draw_pretty(image_np, output_path=vis_path)
        if upload:
            api.annotation.upload_ann(image_id, ann)

    def _predict_image_np(self, image_np: np.ndarray, settings: Optional[str]) -> Annotation:
        """Predict a single image as numpy array.
        
        :param image_np: Image as numpy array
        :type image_np: np.ndarray
        :param settings: Prediction settings string or path to settings file
        :type settings: Optional[str]
        :return: Annotation with prediction results
        :rtype: Annotation
        """
        anns, _ = self.inference._inference_auto([image_np], settings)
        if len(anns) == 0:
            return Annotation(img_size=image_np.shape[:2])
        ann = anns[0]
        return ann

    def predict_local_data_by_args(
        self,
        input_path: str,
        settings: Optional[str] = None,
        output_dir: str = "./predictions",
        draw: bool = False,
    ) -> None:
        """Predict local files or directories.
        
        :param input_path: Path to file or directory
        :type input_path: str
        :param settings: Prediction settings string or path to settings file
        :type settings: Optional[str]
        :param output_dir: Directory for saving results
        :type output_dir: str
        :param draw: Flag for visualization of predictions
        :type draw: bool
        :raises ValueError: If path does not exist or has unsupported format
        """
        logger.info(f"Predicting '{input_path}'")

        if os.path.isdir(input_path):
            self._process_directory(input_path, output_dir, settings, draw)
        elif os.path.isfile(input_path):
            self._process_file(input_path, output_dir, settings, draw)
        else:
            raise ValueError(f"Invalid input path: '{input_path}'")

    def _process_directory(self, input_path: str, output_dir: str, settings: Optional[str], draw: bool) -> None:
        """Process a directory as input.
        
        :param input_path: Path to directory
        :type input_path: str
        :param output_dir: Directory for saving results
        :type output_dir: str
        :param settings: Prediction settings string or path to settings file
        :type settings: Optional[str]
        :param draw: Flag for visualization of predictions
        :type draw: bool
        """
        # Check if directory is a Supervisely project
        project = self._try_load_as_project(input_path)
        
        if project is not None:
            self._process_project(project, output_dir, settings, draw)
        else:
            self._process_regular_directory(input_path, output_dir, settings, draw)

    def _try_load_as_project(self, input_path: str) -> Optional[Union[Project, VideoProject]]:
        """Try to load a directory as a Supervisely project.
        
        :param input_path: Path to directory
        :type input_path: str
        :return: Supervisely project object or None if the directory is not a project
        :rtype: Optional[Union[Project, VideoProject]]
        """
        project = None
        for proj_cls, modality, proj_type in [
            (Project, ProjectType.IMAGES.value, ProjectType.IMAGES),
            (VideoProject, ProjectType.VIDEOS.value, ProjectType.VIDEOS),
        ]:
            try:
                project = proj_cls(input_path, mode=OpenMode.READ)
                logger.info(
                    f"Input directory is a Supervisely {modality} project. The output will be a {modality} project."
                )
                return project
            except Exception as e:
                logger.debug(f"Failed to load as {modality} project: {e}")
        return None

    def _process_project(
        self, 
        project: Union[Project, VideoProject], 
        output_dir: str, 
        settings: Optional[str], 
        draw: bool
    ) -> None:
        """Process a Supervisely project.
        
        :param project: Supervisely project object
        :type project: Union[Project, VideoProject]
        :param output_dir: Directory for saving results
        :type output_dir: str
        :param settings: Prediction settings string or path to settings file
        :type settings: Optional[str]
        :param draw: Flag for visualization of predictions
        :type draw: bool
        """
        proj_type = ProjectType.IMAGES if isinstance(project, Project) else ProjectType.VIDEOS
        output_project = self._create_project(output_dir, project.name, project_type=proj_type)
        output_datasets = {}
        
        generator = (
            self.inference._inference_video_project_generator
            if isinstance(project, VideoProject)
            else self.inference._inference_image_project_generator
        )
        
        postprocess_fn = (
            self._postprocess_video_for_project
            if isinstance(project, VideoProject)
            else self._postprocess_image_for_project
        )
        
        for results in generator(project, settings):
            item_name, item_path, dataset_path = (
                results["item_name"],
                results["item_path"],
                results["dataset_path"],
            )
            ann = (
                [r["annotation"] for r in results["results"]]
                if isinstance(project, VideoProject)
                else results["annotation"]
            )
            dataset_key = dataset_path
            if dataset_key not in output_datasets:
                output_datasets[dataset_key] = output_project.create_dataset(dataset_path)
            postprocess_fn(output_datasets[dataset_key], item_name, item_path, ann, draw=draw)
        
        logger.info(f"Inference results saved to: '{os.path.join(output_dir, output_project.name)}'")

    def _process_regular_directory(
        self, input_path: str, output_dir: str, settings: Optional[str], draw: bool
    ) -> None:
        """Process a regular directory (not a Supervisely project).
        
        :param input_path: Path to directory
        :type input_path: str
        :param output_dir: Directory for saving results
        :type output_dir: str
        :param settings: Prediction settings string or path to settings file
        :type settings: Optional[str]
        :param draw: Flag for visualization of predictions
        :type draw: bool
        :raises ValueError: If directory contains both images and videos
        """
        images = list_files_recursively(str(input_path), sly_image.SUPPORTED_IMG_EXTS, None, True)
        videos = list_files_recursively(str(input_path), sly_video.ALLOWED_VIDEO_EXTENSIONS, None, True)
        
        if images and videos:
            raise ValueError(f"Directory '{input_path}' contains both images and videos.")
        if not images and not videos:
            logger.info("No videos or images found")
            return

        modality = ProjectType.IMAGES.value if images else ProjectType.VIDEOS.value
        project_name = f"{os.path.basename(input_path)}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}"
        in_ds_map, out_ds_map = self._build_dataset_map_for_project(str(input_path), modality, output_dir)
        output_project_path = os.path.join(output_dir, project_name)
        self._create_dataset_structure(output_project_path, in_ds_map, modality)
        self._create_project_files(output_project_path, modality)
        self._process_items_with_dataset_maps(in_ds_map, out_ds_map, settings, modality, draw, output_dir)
        
        logger.info(f"Inference results saved to: '{output_project_path}'")

    def _process_file(
        self, input_path: str, output_dir: str, settings: Optional[str], draw: bool
    ) -> None:
        """Process a single file.
        
        :param input_path: Path to file
        :type input_path: str
        :param output_dir: Directory for saving results
        :type output_dir: str
        :param settings: Prediction settings string or path to settings file
        :type settings: Optional[str]
        :param draw: Flag for visualization of predictions
        :type draw: bool
        :raises ValueError: If file format is not supported
        """
        input_path_str = str(input_path)
        if self._is_image(input_path_str):
            try:
                image_np = sly_image.read(input_path_str)
                anns, _ = self.inference._inference_auto([image_np], settings)
                self._postprocess_image(input_path_str, anns[0], None, False, draw, output_dir)
                logger.info(f"Inference results saved to: '{output_dir}'")
            except Exception as e:
                logger.error(f"Failed to process image '{input_path_str}': {e}")
        elif self._is_video(input_path_str):
            try:
                anns = [i["annotation"] for i in self.inference._inference_video_path(input_path_str)]
                self._postprocess_video(input_path_str, anns, None, False, draw, output_dir)
                logger.info(f"Inference results saved to: '{output_dir}'")
            except Exception as e:
                logger.error(f"Failed to process video '{input_path_str}': {e}")
        else:
            raise ValueError(f"Unsupported input format: '{input_path}'. Expected image or directory.")

    def _create_project(
        self, 
        output_dir: str, 
        project_name: str, 
        project_type: ProjectType = ProjectType.IMAGES
    ) -> Union[Project, VideoProject]:
        """Create a new Supervisely project.
        
        :param output_dir: Directory to create project in
        :type output_dir: str
        :param project_name: Project name
        :type project_name: str
        :param project_type: Project type (images or videos)
        :type project_type: ProjectType
        :return: Created Supervisely project
        :rtype: Union[Project, VideoProject]
        """
        project_name = f'{project_name}_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}'
        project_cls = Project if project_type == ProjectType.IMAGES else VideoProject
        project = project_cls(f"{output_dir}/{project_name}", mode=OpenMode.CREATE)
        project.set_meta(self.inference.model_meta)
        return project

    def _create_project_files(self, output_project_path: str, modality: str) -> None:
        """Create necessary files for a Supervisely project.
        
        :param output_project_path: Project directory path
        :type output_project_path: str
        :param modality: Project type ('images' or 'videos')
        :type modality: str
        """
        output_project_meta_path = os.path.join(output_project_path, "meta.json")
        ensure_base_path(output_project_meta_path)
        dump_json_file(self.inference.model_meta.to_json(), output_project_meta_path)
        if modality == ProjectType.VIDEOS.value:
            output_key_id_map_path = os.path.join(output_project_path, "key_id_map.json")
            key_id_map = KeyIdMap()
            key_id_map.dump_json(output_key_id_map_path)

    def _create_dataset_structure(self, base_path: str, dataset_map: Dict, modality: str) -> None:
        """Create directory structure based on dataset_map with modality-specific subdirectories.
        
        :param base_path: Base path
        :type base_path: str
        :param dataset_map: Dictionary describing dataset structure
        :type dataset_map: Dict
        :param modality: Project type ('images' or 'videos')
        :type modality: str
        """
        modality_dir = "img" if modality == ProjectType.IMAGES.value else "video"
        for ds_name, ds_content in dataset_map.items():
            ds_path = os.path.join(base_path, ds_name)
            os.makedirs(os.path.join(ds_path, modality_dir), exist_ok=True)
            os.makedirs(os.path.join(ds_path, "ann"), exist_ok=True)
            if ds_content.get("datasets"):
                nested_path = os.path.join(ds_path, "datasets")
                os.makedirs(nested_path, exist_ok=True)
                self._create_dataset_structure(nested_path, ds_content["datasets"], modality)

    def _postprocess_image(
        self, 
        image_path: str, 
        ann: Annotation, 
        output_image_path: Optional[str] = None, 
        copy_item: bool = False,
        draw: bool = False,
        output_dir: str = "./predictions"
    ) -> None:
        """Save image prediction results.
        
        :param image_path: Path to image
        :type image_path: str
        :param ann: Annotation with prediction results
        :type ann: Annotation
        :param output_image_path: Path to save image (if None, original path is used)
        :type output_image_path: Optional[str]
        :param copy_item: Flag to copy original image
        :type copy_item: bool
        :param draw: Flag for visualization of predictions
        :type draw: bool
        :param output_dir: Directory for saving results
        :type output_dir: str
        """
        image_name = get_file_name_with_ext(image_path)
        pred_ann_path = (
            os.path.join(output_dir, f"{image_name}.json")
            if not output_image_path
            else os.path.join(
                os.path.dirname(os.path.dirname(output_image_path)),
                "ann",
                f"{image_name}.json",
            )
        )
        if output_image_path and copy_item:
            copy_file(image_path, output_image_path)
        ensure_base_path(pred_ann_path)
        dump_json_file(ann.to_json(), pred_ann_path)
        if draw:
            preview_path = (
                os.path.join(output_dir, "previews", image_name)
                if not output_image_path
                else os.path.join(
                    os.path.dirname(os.path.dirname(output_image_path)),
                    "preview",
                    image_name,
                )
            )
            ensure_base_path(preview_path)
            image = sly_image.read(image_path)
            ann.draw_pretty(image, output_path=preview_path)

    def _postprocess_video(
        self, 
        video_path: str, 
        anns: List[Any], 
        output_video_path: Optional[str] = None, 
        copy_item: bool = False,
        draw: bool = False,
        output_dir: str = "./predictions"
    ) -> None:
        """Save video prediction results.
        
        :param video_path: Path to video
        :type video_path: str
        :param anns: List of annotations with prediction results for each frame
        :type anns: List[Any]
        :param output_video_path: Path to save video (if None, original path is used)
        :type output_video_path: Optional[str]
        :param copy_item: Flag to copy original video
        :type copy_item: bool
        :param draw: Flag for visualization of predictions
        :type draw: bool
        :param output_dir: Directory for saving results
        :type output_dir: str
        """
        video_ann = self._create_video_ann(video_path, anns)
        video_name = get_file_name_with_ext(video_path)
        pred_ann_path = (
            os.path.join(output_dir, f"{video_name}.json")
            if not output_video_path
            else os.path.join(
                os.path.dirname(os.path.dirname(output_video_path)),
                "ann",
                f"{video_name}.json",
            )
        )
        if output_video_path and copy_item:
            copy_file(video_path, output_video_path)
        ensure_base_path(pred_ann_path)
        dump_json_file(video_ann.to_json(), pred_ann_path)

    def _postprocess_image_for_project(
        self,
        dataset: Dataset,
        image_name: str,
        image_path: str,
        ann: Union[Annotation, Dict[str, Any]],
        draw: bool = False,
        output_dir: str = "./predictions",
        copy_item: bool = False
    ) -> None:
        """Add image prediction to a Supervisely project.
        
        :param dataset: Supervisely dataset
        :type dataset: Dataset
        :param image_name: Image name
        :type image_name: str
        :param image_path: Path to image
        :type image_path: str
        :param ann: Annotation with prediction results or dictionary with annotation
        :type ann: Union[Annotation, Dict[str, Any]]
        :param draw: Flag for visualization of predictions
        :type draw: bool
        :param output_dir: Directory for saving results
        :type output_dir: str
        :param copy_item: Flag to copy original image
        :type copy_item: bool
        """
        dataset.add_item_file(item_name=image_name, item_path=None, ann=ann)
        if draw:
            preview_path = os.path.join(output_dir, "previews", image_name)
            ensure_base_path(preview_path)
            image = sly_image.read(image_path)
            ann.draw_pretty(image, output_path=preview_path)
        if copy_item:
            copy_file(image_path, os.path.join(dataset.directory, "img", image_name))

    def _postprocess_video_for_project(
        self,
        dataset: VideoDataset,
        video_name: str,
        video_path: str,
        anns: Union[List[Dict[str, Any]], List[Annotation]],
        draw: bool = False,
        output_dir: str = "./predictions",
        copy_item: bool = False
    ) -> None:
        """Add video prediction to a Supervisely project.
        
        :param dataset: Supervisely video dataset
        :type dataset: VideoDataset
        :param video_name: Video name
        :type video_name: str
        :param video_path: Path to video
        :type video_path: str
        :param anns: List of annotations with prediction results or list of dictionaries with annotations
        :type anns: Union[List[Dict[str, Any]], List[Annotation]]
        :param draw: Flag for visualization of predictions
        :type draw: bool
        :param output_dir: Directory for saving results
        :type output_dir: str
        :param copy_item: Flag to copy original video
        :type copy_item: bool
        """
        video_ann = self._create_video_ann(video_path, anns)
        dataset.add_item_file(item_name=video_name, item_path=None, ann=video_ann)
        if copy_item:
            copy_file(video_path, os.path.join(dataset.directory, "video", video_name))

    def _create_video_ann(
        self, video_path: str, anns: Union[List[Dict[str, Any]], List[Annotation]]
    ) -> VideoAnnotation:
        """Create a VideoAnnotation from a list of annotations.
        
        :param video_path: Path to video
        :type video_path: str
        :param anns: List of annotations or dictionaries with annotations for each frame
        :type anns: Union[List[Dict[str, Any]], List[Annotation]]
        :return: VideoAnnotation object with annotations
        :rtype: VideoAnnotation
        :raises RuntimeError: If video cannot be opened
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        video_objects, video_frames = [], []
        for frame_idx, ann in enumerate(anns):
            if not isinstance(ann, Annotation):
                ann = Annotation.from_json(ann, self.inference.model_meta)
            frame_figures = []
            for label in ann.labels:
                tags = VideoTagCollection(
                    [VideoTag(meta=tag.meta, value=tag.value) for tag in label.tags]
                )
                obj = VideoObject(obj_class=label.obj_class, tags=tags)
                video_objects.append(obj)
                frame_figures.append(
                    VideoFigure(
                        video_object=obj,
                        geometry=label.geometry,
                        frame_index=frame_idx,
                    )
                )
            video_frames.append(Frame(frame_idx, figures=frame_figures))
        return VideoAnnotation(
            img_size=(frame_height, frame_width),
            frames_count=frames_count,
            objects=VideoObjectCollection(video_objects),
            frames=FrameCollection(video_frames),
        )

    def _is_image(self, path: str) -> bool:
        """Check if path is an image.
        
        :param path: Path to file
        :type path: str
        :return: True if file is an image, False otherwise
        :rtype: bool
        """
        return path.lower().endswith(tuple(ext.lower() for ext in sly_image.SUPPORTED_IMG_EXTS))

    def _is_video(self, path: str) -> bool:
        """Check if path is a video.
        
        :param path: Path to file
        :type path: str
        :return: True if file is a video, False otherwise
        :rtype: bool
        """
        return path.lower().endswith(tuple(ext.lower() for ext in sly_video.ALLOWED_VIDEO_EXTENSIONS))

    def _build_dataset_map_for_project(
        self, project_path: str, modality: str, output_dir: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build input and output dataset maps for a project directory.
        
        :param project_path: Path to project directory
        :type project_path: str
        :param modality: Project type ('images' or 'videos')
        :type modality: str
        :param output_dir: Directory for saving results
        :type output_dir: str
        :return: Tuple (input data map, output data map)
        :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
        :raises ValueError: If project type is not supported
        """
        if modality not in (ProjectType.IMAGES.value, ProjectType.VIDEOS.value):
            raise ValueError(f"Unsupported project modality: {modality}")
        
        modality_dir = "img" if modality == ProjectType.IMAGES.value else "video"
        project_name = f"{os.path.basename(project_path)}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}"
        
        in_ds_map, out_ds_map = {}, {}
        
        # Process root files
        root_files = [f for f in os.listdir(project_path) if os.path.isfile(os.path.join(project_path, f))]
        root_items = []
        if modality == ProjectType.IMAGES.value:
            root_items = [f for f in root_files if self._is_image(f)]
        elif modality == ProjectType.VIDEOS.value:
            root_items = [f for f in root_files if self._is_video(f)]
        
        subdirs = [d for d in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, d))]
        if root_items:
            base_name = 'root_files'
            unique_name = base_name
            counter = 1
            # Ensure the dataset name is unique by incrementing if necessary
            while unique_name in subdirs:
                unique_name = f"{base_name}{counter}"
                counter += 1
            in_ds_map[unique_name] = {"datasets": {}, "items": [os.path.join(project_path, item) for item in root_items]}
            out_ds_map[unique_name] = {"datasets": {}, "items": [os.path.join(output_dir, project_name, unique_name, modality_dir, item) for item in root_items]}
        
        # Process subdirectories
        for entry in subdirs:
            full_path = os.path.join(project_path, entry)
            in_path = os.path.join(project_path, entry)
            out_path = os.path.join(output_dir, project_name, entry)
            in_res, out_res = self._build_dataset_map(
                full_path, in_path, out_path, modality_dir
            )
            in_ds_map[entry] = in_res
            out_ds_map[entry] = out_res
        
        return in_ds_map, out_ds_map

    def _build_dataset_map(
        self, 
        directory_path: str, 
        current_in_path: str, 
        current_out_path: str, 
        modality_dir: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Recursively build dataset maps.
        
        :param directory_path: Path to directory
        :type directory_path: str
        :param current_in_path: Current input path
        :type current_in_path: str
        :param current_out_path: Current output path
        :type current_out_path: str
        :param modality_dir: Subdirectory for data type ('img' or 'video')
        :type modality_dir: str
        :return: Tuple (input data map, output data map)
        :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
        """
        in_ds_map = {"datasets": {}, "items": []}
        out_ds_map = {"datasets": {}, "items": []}
        
        try:
            entries = os.listdir(directory_path)
        except OSError as e:
            logger.warning(f"Failed to list directory {directory_path}: {e}")
            return in_ds_map, out_ds_map
        
        for entry in entries:
            full_path = os.path.join(directory_path, entry)
            if os.path.isdir(full_path):
                in_sub, out_sub = self._build_dataset_map(
                    full_path,
                    f"{current_in_path}/{entry}",
                    f"{current_out_path}/datasets/{entry}",
                    modality_dir,
                )
                in_ds_map["datasets"][entry] = in_sub
                out_ds_map["datasets"][entry] = out_sub
            elif os.path.isfile(full_path):
                in_ds_map["items"].append(f"{current_in_path}/{entry}")
                out_ds_map["items"].append(f"{current_out_path}/{modality_dir}/{entry}")
        
        return in_ds_map, out_ds_map

    def _process_items_with_dataset_maps(
        self, 
        in_ds_map: Dict[str, Any], 
        out_ds_map: Dict[str, Any], 
        settings: Optional[str], 
        modality: str, 
        draw: bool, 
        output_dir: str
    ) -> None:
        """Process items from in_ds_map and save to out_ds_map.
        
        :param in_ds_map: Input data map
        :type in_ds_map: Dict[str, Any]
        :param out_ds_map: Output data map
        :type out_ds_map: Dict[str, Any]
        :param settings: Prediction settings string or path to settings file
        :type settings: Optional[str]
        :param modality: Project type ('images' or 'videos')
        :type modality: str
        :param draw: Flag for visualization of predictions
        :type draw: bool
        :param output_dir: Directory for saving results
        :type output_dir: str
        """
        item_pairs = self._collect_item_pairs(in_ds_map, out_ds_map)
        
        if not item_pairs:
            logger.info("No items found in datasets")
            return

        if modality == ProjectType.IMAGES.value:
            # Process images in batch
            input_images = [in_path for in_path, _ in item_pairs]
            try:
                anns, _ = self.inference._inference_auto(input_images, settings)
                for (in_path, out_path), ann in zip(item_pairs, anns):
                    self._postprocess_image(in_path, ann, out_path, True, draw, output_dir)
            except Exception as e:
                logger.error(f"Image inference failed: {e}")
        else:
            # Process videos one by one
            for in_path, out_path in item_pairs:
                try:
                    anns = [i["annotation"] for i in self.inference._inference_video_path(in_path)]
                    self._postprocess_video(in_path, anns, out_path, True, draw, output_dir)
                except Exception as e:
                    logger.error(f"Video inference failed for '{in_path}': {e}")

    def _collect_item_pairs(
        self, in_map: Dict[str, Any], out_map: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """Recursively collect input-output path pairs.
        
        :param in_map: Input data map
        :type in_map: Dict[str, Any]
        :param out_map: Output data map
        :type out_map: Dict[str, Any]
        :return: List of tuples (input path, output path)
        :rtype: List[Tuple[str, str]]
        :raises ValueError: If dataset names or item counts don't match
        """
        item_pairs = []
        for ds_name in in_map:
            if ds_name not in out_map:
                raise ValueError(f"Dataset '{ds_name}' not found in output map")
            in_items = in_map[ds_name].get("items", [])
            out_items = out_map[ds_name].get("items", [])
            if len(in_items) != len(out_items):
                raise ValueError(
                    f"Mismatch in item counts for dataset '{ds_name}': {len(in_items)} vs {len(out_items)}"
                )
            item_pairs.extend(zip(in_items, out_items))
            sub_pairs = self._collect_item_pairs(
                in_map[ds_name].get("datasets", {}),
                out_map[ds_name].get("datasets", {}),
            )
            item_pairs.extend(sub_pairs)
        return item_pairs 