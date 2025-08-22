from __future__ import annotations

import os
import math
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, List
from urllib.parse import urlencode

import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
from supervisely import logger, ProjectInfo
from supervisely.api.api import Api
from supervisely.api.file_api import FileInfo
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.cuboid import Cuboid
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.geometry.point import Point
from supervisely.geometry.point_3d import Point3d
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging.color import rgb2hex
from supervisely.nn.benchmark.object_detection.metric_provider import (
    METRIC_NAMES as OBJECT_DETECTION_METRIC_NAMES,
)
from supervisely.nn.benchmark.semantic_segmentation.metric_provider import (
    METRIC_NAMES as SEMANTIC_SEGMENTATION_METRIC_NAMES,
)
from supervisely.nn.inference import Inference
from supervisely.nn.task_type import TaskType
from supervisely.nn.utils import RuntimeType
from supervisely.project import ProjectMeta
from supervisely.template.base_generator import BaseGenerator

try:
    from tbparse import SummaryReader  # pylint: disable=import-error
    import plotly.express as px  # pylint: disable=import-error
    from plotly.subplots import make_subplots  # pylint: disable=import-error
    import plotly.graph_objects as go  # pylint: disable=import-error
except Exception as _:
    SummaryReader = None  # type: ignore
    px = None  # type: ignore
    make_subplots = None  # type: ignore
    go = None  # type: ignore

# @TODO: Partly supports unreleased apps
class ExperimentGenerator(BaseGenerator):

    def __init__(
        self,
        api: Api,
        experiment_info: dict,
        hyperparameters: str,
        model_meta: ProjectMeta,
        serving_class: Optional[Inference] = None,
        team_id: Optional[int] = None,
        output_dir: str = "./experiment_report",
        app_options: Optional[dict] = None,
    ):
        """Initialize experiment generator class.

        :param api: Supervisely API instance
        :type api: Api
        :param experiment_info: Dictionary with experiment information
        :type experiment_info: Dict[str, Any]
        :param hyperparameters: Hyperparameters as YAML string or dictionary
        :type hyperparameters: Optional[Union[str, Dict]]
        :param model_meta: Model metadata as dictionary
        :type model_meta: Optional[Union[str, Dict]]
        :param serving_class: Serving class for model inference
        :type serving_class: Optional[Inference]
        """
        super().__init__(api, output_dir=output_dir)
        self.team_id = team_id or sly_env.team_id()
        self.info = experiment_info
        self.hyperparameters = hyperparameters
        self.model_meta = model_meta
        self.artifacts_dir = self.info["artifacts_dir"]
        self.serving_class = serving_class
        self.app_info = self._get_app_info()
        self.app_options = app_options

    def _report_url(self, server_address: str, template_id: int) -> str:
        return f"{server_address}/nn/experiments/{template_id}"

    def _datasets_url_with_entities_filter(self, project_id: int, entities_filter: List[dict]) -> str:
        base_url = self.api.server_address.rstrip('/')
        path = f"/projects/{project_id}/datasets"
        query = urlencode({"entitiesFilter": json.dumps(entities_filter)})
        return f"{base_url}{path}?{query}"

    def upload_to_artifacts(self):
        remote_dir = os.path.join(self.info["artifacts_dir"], "visualization")
        self.upload(remote_dir, team_id=self.team_id)

    def get_report(self) -> FileInfo:
        remote_report_path = os.path.join(
            self.info["artifacts_dir"], "visualization", "template.vue"
        )
        experiment_report = self.api.file.get_info_by_path(self.team_id, remote_report_path)
        if experiment_report is None:
            raise ValueError("Generate and upload report first")
        return experiment_report

    def get_report_id(self) -> int:
        return self.get_report().id

    def get_report_link(self) -> str:
        return self._report_url(self.api.server_address, self.get_report_id())

    def state(self) -> dict:
        return {}

    def context(self) -> dict:
        context = {
            "env": self._get_env_context(),
            "experiment": self._get_experiment_context(),
            "resources": self._get_resources_context(),
            "code": self._get_code_context(),
            "widgets": self._get_widgets_context(),
        }
        return context

    # --------------------------- Context blocks helpers --------------------------- #
    def _get_env_context(self):
        return {"server_address": self.api.server_address}

    def _get_apps_context(self):
        train_app, serve_app = self._get_app_train_serve_app_info()
        log_viewer_app = self._get_log_viewer_app_info()
        apply_images_app, apply_videos_app = self._get_app_apply_nn_app_info()
        predict_app = self._get_predict_app_info()
        return {
            "train": train_app,
            "serve": serve_app,
            "log_viewer": log_viewer_app,
            "apply_nn_to_images": apply_images_app,
            "apply_nn_to_videos": apply_videos_app,
            "predict": predict_app,
        }

    def _get_original_repository_info(self):
        original_repository = self.app_options.get("original_repository", None)
        if original_repository is None:
            return None
        original_repository_info = {
            "name": original_repository.get("name", None),
            "url": original_repository.get("url", None),
        }
        return original_repository_info

    def _get_resources_context(self):
        apps = self._get_apps_context()
        original_repository = self._get_original_repository_info()
        return {"apps": apps, "original_repository": original_repository}

    def _get_code_context(self):
        docker_image = self._get_docker_image()
        repo_info = self._get_repository_info()
        pytorch_demo, onnx_demo, trt_demo = self._get_demo_scripts()

        return {
            "docker": {"image": docker_image, "deploy": f"{docker_image}-deploy"},
            "local_prediction": {
                "repo": repo_info,
                "serving_module": (self.serving_class.__module__ if self.serving_class else None),
                "serving_class": (self.serving_class.__name__ if self.serving_class else None),
            },
            "demo": {
                "pytorch": pytorch_demo,
                "onnx": onnx_demo,
                "tensorrt": trt_demo,
            },
        }

    def _get_widgets_context(self):
        checkpoints_table = self._generate_checkpoints_table()
        metrics_table = self._generate_metrics_table(self.info["task_type"])
        sample_gallery = self._get_sample_predictions_gallery()
        classes_table = self._generate_classes_table()
        training_plots = self._generate_training_plots()
        return {
            "tables": {
                "checkpoints": checkpoints_table,
                "metrics": metrics_table,
                "classes": classes_table,
            },
            "sample_pred_gallery": sample_gallery,
            "training_plots": training_plots,
        }

    # --------------------------------------------------------------------------- #

    def _generate_metrics_table(self, task_type: str) -> str:
        """Generate HTML table with evaluation metrics.

        :returns: HTML string with metrics table
        :rtype: str
        """
        metrics = self.info.get("evaluation_metrics", {})
        if not metrics:
            return None

        html = ['<table class="table">']
        html.append("<thead><tr><th>Metrics</th><th>Value</th></tr></thead>")
        html.append("<tbody>")

        if task_type == TaskType.OBJECT_DETECTION or task_type == TaskType.INSTANCE_SEGMENTATION:
            metric_names = OBJECT_DETECTION_METRIC_NAMES
        elif task_type == TaskType.SEMANTIC_SEGMENTATION:
            metric_names = SEMANTIC_SEGMENTATION_METRIC_NAMES
        else:
            raise NotImplementedError(f"Task type '{task_type}' is not supported")

        for metric_name, metric_value in metrics.items():
            formatted_metric_name = metric_names.get(metric_name)
            if formatted_metric_name is None:
                formatted_metric_name = metric_name.replace("_", " ")
                formatted_metric_name = formatted_metric_name.replace("-", " ")
            if isinstance(metric_value, float):
                metric_value = f"{metric_value:.4f}"
            html.append(f"<tr><td>{metric_name}</td><td>{metric_value}</td></tr>")

        html.append("</tbody>")
        html.append("</table>")
        return "\n".join(html)

    def _generate_classes_table(self) -> str:
        """Generate HTML table with class names, shapes and colors.

        :returns: HTML string with classes table
        :rtype: str
        """
        type_to_icon = {
            AnyGeometry: "zmdi zmdi-shape",
            Rectangle: "zmdi zmdi-crop-din",
            Polygon: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAB6klEQVRYhe2Wuy8EURTGf+u5VESNXq2yhYZCoeBv8RcI1i6NVUpsoVCKkHjUGlFTiYb1mFmh2MiKjVXMudmb3cPOzB0VXzKZm5k53/nmvO6Ff4RHD5AD7gFP1l3Kd11AHvCBEpAVW2esAvWmK6t8l1O+W0lCQEnIJoAZxUnzNQNkZF36jrQjgoA+uaciCgc9VaExBOyh/6WWAi1VhbjOJ4FbIXkBtgkK0BNHnYqNKUIPeBPbKyDdzpld5T6wD9SE4AwYjfEDaXFeFzE/doUWuhqwiFsOCwqv2hV2lU/L+sHBscGTxdvSFVoXpAjCZdauMHVic6ndl6U1VBsJCFhTeNUU9IiIEo3qvQYGHAV0AyfC5wNLhKipXuBCjA5wT8WxcM1FMRoBymK44CjAE57hqIazwCfwQdARcXa3UXHuRXVucIjb7jYvNkdxBZg0TBFid7PQTRAtX2xOiXkuMAMqYwkIE848rZFbjyNAmw9bIeweaZ2A5TgC7PnwKkTPtN+cTOrsyN3FEWAjRTAX6sA5ek77gSL6+WHZVQDAIHAjhJtN78aAS3lXAXYIivBOnCdyOAUYB6o0xqsvziry7FLE/Cp20cNcJEjDr8MUmVOVRzkVN+Nd7vZGVXXgiwxtPiRS5WFhz4fEq/zv4AvToMn7vCn3eAAAAABJRU5ErkJggg==",
            Bitmap: "zmdi zmdi-brush",
            Polyline: "zmdi zmdi-gesture",
            Point: "zmdi zmdi-dot-circle-alt",
            Cuboid: "zmdi zmdi-ungroup",  #
            GraphNodes: "zmdi zmdi-grain",
            Cuboid3d: "zmdi zmdi-codepen",
            Pointcloud: "zmdi zmdi-cloud-outline",
            MultichannelBitmap: "zmdi zmdi-layers",
            Point3d: "zmdi zmdi-filter-center-focus",
        }

        if not hasattr(self.model_meta, "obj_classes"):
            return None

        if len(self.model_meta.obj_classes) == 0:
            return None

        html = ['<table class="table">']
        html.append("<thead><tr><th>Class name</th><th>Shape</th></tr></thead>")
        html.append("<tbody>")

        for obj_class in self.model_meta.obj_classes:
            class_name = obj_class.name
            color_hex = rgb2hex(obj_class.color)
            icon = type_to_icon.get(obj_class.geometry_type, "zmdi zmdi-shape")

            class_cell = (
                f"<i class='zmdi zmdi-circle' style='color: {color_hex}; margin-right: 5px;'></i>"
                f"<span>{class_name}</span>"
            )

            if isinstance(icon, str) and icon.startswith("data:image"):
                shape_cell = f"<img src='{icon}' style='height: 15px; margin-right: 2px;'/>"
            else:
                shape_cell = f"<i class='{icon}' style='margin-right: 5px;'></i>"

            shape_name = obj_class.geometry_type.geometry_name()
            shape_cell += f"<span>{shape_name}</span>"

            html.append(f"<tr><td>{class_cell}</td><td>{shape_cell}</td></tr>")

        html.append("</tbody>")
        html.append("</table>")
        return "\n".join(html)

    def _generate_metrics_table_horizontal(self) -> str:
        """Generate HTML table with evaluation metrics.

        :returns: HTML string with metrics table
        :rtype: str
        """
        metrics = self.info.get("evaluation_metrics", {})
        if not metrics:
            return None

        html = ['<table class="table">']
        # Generate header row with metric names
        header_cells = []
        for metric_name in metrics.keys():
            metric_name = metric_name.replace("_", " ")
            metric_name = metric_name.replace("-", " ")
            header_cells.append(f"<th>{metric_name}</th>")
        html.append(f"<thead><tr>{''.join(header_cells)}</tr></thead>")

        # Generate value row
        html.append("<tbody>")
        value_cells = []
        for metric_value in metrics.values():
            if isinstance(metric_value, float):
                metric_value = f"{metric_value:.4f}"
            value_cells.append(f"<td>{metric_value}</td>")
        html.append(f"<tr>{''.join(value_cells)}</tr>")
        html.append("</tbody>")

        html.append("</table>")
        return "\n".join(html)

    def _get_primary_metric(self) -> dict:
        """Get primary metric from evaluation metrics.

        :returns: Primary metric info
        :rtype: dict
        """
        primary_metric = {
            "name": None,
            "value": None,
            "rounded_value": None,
        }

        eval_metrics = self.info.get("evaluation_metrics", {})
        primary_metric_name = self.info.get("primary_metric")
        primary_metric_value = eval_metrics.get(primary_metric_name)

        if primary_metric_name is None or primary_metric_value is None:
            logger.debug(f"Primary metric is not found in evaluation metrics: {eval_metrics}")
            return primary_metric

        primary_metric = {
            "name": primary_metric_name,
            "value": primary_metric_value,
            "rounded_value": round(primary_metric_value, 3),
        }
        return primary_metric

    def _get_display_metrics(self, task_type: str) -> list:
        """Get first 5 metrics for display (excluding primary metric).

        :param primary_metric: Primary metric info
        :type primary_metric: dict
        :returns: List of tuples (metric_name, metric_value) for display
        :rtype: list
        """
        display_metrics = []
        eval_metrics = self.info.get("evaluation_metrics", {})
        if not eval_metrics:
            return display_metrics

        main_metrics = []
        if task_type == TaskType.OBJECT_DETECTION or task_type == TaskType.INSTANCE_SEGMENTATION:
            main_metrics = ["mAP", "AP75", "AP50", "precision", "recall"]
        elif task_type == TaskType.SEMANTIC_SEGMENTATION:
            main_metrics = ["mIoU", "mPixel", "mPrecision", "mRecall", "mF1"]
        else:
            raise NotImplementedError(f"Task type '{task_type}' is not supported")

        for metric_name in main_metrics:
            if metric_name in eval_metrics:
                metric_value = eval_metrics[metric_name]
                value = round(metric_value, 3)
                percent_value = round(metric_value * 100, 3)
                display_metrics.append(
                    {
                        "name": metric_name,
                        "value": value,
                        "percent_value": percent_value,
                    }
                )

        return display_metrics

    def _generate_checkpoints_table(self) -> str:
        """Generate HTML table with checkpoint information.

        :returns: HTML string with checkpoints table
        :rtype: str
        """
        pytorch_checkpoints = self.info.get("checkpoints", None)
        if pytorch_checkpoints is None:
            raise ValueError("Checkpoints are not found in experiment info")

        checkpoints = pytorch_checkpoints.copy()
        export = self.info.get("export", {})
        if export:
            onnx_checkpoint = export.get(RuntimeType.ONNXRUNTIME)
            trt_checkpoint = export.get(RuntimeType.TENSORRT)
            if onnx_checkpoint is not None:
                checkpoints.append(onnx_checkpoint)
            if trt_checkpoint is not None:
                checkpoints.append(trt_checkpoint)

        checkpoint_paths = [os.path.join(self.artifacts_dir, ckpt) for ckpt in checkpoints]
        checkpoint_infos = [
            self.api.file.get_info_by_path(self.team_id, path) for path in checkpoint_paths
        ]
        checkpoint_sizes = [f"{info.sizeb / 1024 / 1024:.2f} MB" for info in checkpoint_infos]
        checkpoint_dl_links = [
            f"<a href='{info.full_storage_url}' download='{sly_fs.get_file_name_with_ext(info.path)}'>Download</a>"
            for info in checkpoint_infos
        ]

        html = ['<table class="table">']
        html.append("<thead><tr><th>Checkpoint</th><th>Size</th><th> </th></tr></thead>")
        html.append("<tbody>")
        for checkpoint, size, dl_link in zip(checkpoints, checkpoint_sizes, checkpoint_dl_links):
            if isinstance(checkpoint, str):
                html.append(
                    f"<tr><td>{os.path.basename(checkpoint)}</td><td>{size}</td><td>{dl_link}</td></tr>"
                )
        html.append("</tbody>")
        html.append("</table>")
        return "\n".join(html)

    def _generate_hyperparameters_yaml(self) -> str:
        """Return hyperparameters as YAML string.

        :returns: YAML string with hyperparameters
        :rtype: str
        """
        if self.hyperparameters is not None:
            if not isinstance(self.hyperparameters, str):
                raise ValueError("Hyperparameters must be a yaml string")
            hyperparameters = self.hyperparameters.split("\n")
            return hyperparameters
        return None

    def _get_log_viewer_app_info(self):
        """Get log viewer app information.

        :returns: Log viewer app info
        :rtype: dict
        """
        slug = "supervisely-ecosystem/tensorboard-experiments-viewer"
        module_id = self.api.app.get_ecosystem_module_id(slug)
        return {"slug": slug, "module_id": module_id}

    def _get_app_info(self):
        """Get app information from task.

        :returns: App info object or None if not found
        :rtype: Optional[Any]
        """
        task_id = self.info.get("task_id", None)
        if task_id is None or task_id == -1:
            return None

        task_info = self.api.task.get_info_by_id(task_id)
        app_id = task_info["meta"]["app"]["id"]
        return self.api.app.get_info_by_id(app_id)

    def _get_training_session(self) -> dict:
        """Get training session information.

        :returns: Training session info
        :rtype: dict
        """
        task_id = self.info["task_id"]
        if task_id is None or task_id == -1:
            training_session = {
                "id": None,
                "url": None,
            }
            return training_session

        training_session = {
            "id": task_id,
            "url": f"{self.api.server_address}/apps/sessions/{task_id}",
        }
        return training_session

    def _get_training_duration(self) -> str:
        """Get training duration.

        :returns: Training duration in format "{h}h {m}m" or "N/A"
        :rtype: str
        """
        raw_duration = self.info.get("training_duration", "N/A")
        if raw_duration in (None, "N/A"):
            return "N/A"

        try:
            duration_seconds = float(raw_duration)
        except (TypeError, ValueError):
            return str(raw_duration)

        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        return f"{hours}h {minutes}m {seconds}s"

    def _get_date(self) -> str:
        """Format experiment date.

        :returns: Formatted date string
        :rtype: str
        """
        date_str = self.info.get("datetime", "")
        date = date_str
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                date = dt.strftime("%d %b %Y")
            except ValueError:
                pass
        return date

    def _get_best_checkpoint(self) -> dict:
        """Get best checkpoint filename.

        :returns: Best checkpoint info
        :rtype: dict
        """
        best_checkpoint_path = os.path.join(
            self.artifacts_dir, "checkpoints", self.info["best_checkpoint"]
        )
        best_checkpoint_info = self.api.file.get_info_by_path(self.team_id, best_checkpoint_path)
        best_checkpoint = {
            "name": self.info["best_checkpoint"],
            "path": best_checkpoint_path,
            "url": best_checkpoint_info.full_storage_url,
            "size": f"{best_checkpoint_info.sizeb / 1024 / 1024:.1f} MB",
        }
        return best_checkpoint

    def _get_optimized_checkpoints(self) -> Tuple[dict, dict]:
        """Get optimized checkpoint filename (ONNX or TensorRT).

        :returns: Checkpoint info or None if not available
        :rtype: Optional[dict]
        """
        export = self.info.get("export", {})

        onnx_checkpoint_data = {
            "name": None,
            "path": None,
            "size": None,
            "url": None,
            "classes_url": None,
        }
        trt_checkpoint_data = {
            "name": None,
            "path": None,
            "size": None,
            "url": None,
            "classes_url": None,
        }

        onnx_checkpoint = export.get(RuntimeType.ONNXRUNTIME)
        if onnx_checkpoint is not None:
            onnx_checkpoint_path = os.path.join(
                self.artifacts_dir, export.get(RuntimeType.ONNXRUNTIME)
            )
            classes_file = self.api.file.get_info_by_path(
                self.team_id,
                os.path.join(os.path.dirname(onnx_checkpoint_path), "classes.json"),
            )
            onnx_file_info = self.api.file.get_info_by_path(self.team_id, onnx_checkpoint_path)
            onnx_checkpoint_data = {
                "name": os.path.basename(export.get(RuntimeType.ONNXRUNTIME)),
                "path": onnx_checkpoint_path,
                "size": f"{onnx_file_info.sizeb / 1024 / 1024:.1f} MB",
                "url": onnx_file_info.full_storage_url,
                "classes_url": classes_file.full_storage_url if classes_file else None,
            }
        trt_checkpoint = export.get(RuntimeType.TENSORRT)
        if trt_checkpoint is not None:
            trt_checkpoint_path = os.path.join(self.artifacts_dir, export.get(RuntimeType.TENSORRT))
            classes_file = self.api.file.get_info_by_path(
                self.team_id,
                os.path.join(os.path.dirname(trt_checkpoint_path), "classes.json"),
            )
            trt_file_info = self.api.file.get_info_by_path(self.team_id, trt_checkpoint_path)
            trt_checkpoint_data = {
                "name": os.path.basename(export.get(RuntimeType.TENSORRT)),
                "path": trt_checkpoint_path,
                "size": f"{trt_file_info.sizeb / 1024 / 1024:.1f} MB",
                "url": trt_file_info.full_storage_url,
                "classes_url": classes_file.full_storage_url if classes_file else None,
            }
        return onnx_checkpoint_data, trt_checkpoint_data

    def _get_docker_image(self) -> str:
        """Get Docker image for model.

        :returns: Docker image name
        :rtype: str
        """
        if self.app_info is None:
            return None

        docker_image = self.app_info.config["docker_image"]
        if not docker_image:
            raise ValueError("Docker image is not found in app config")
        return docker_image

    def _get_repository_info(self) -> Dict[str, str]:
        """Get repository information.

        :returns: Dictionary with repo URL and name
        :rtype: Dict[str, str]
        """
        if self.app_info is None:
            return {"url": None, "name": None}

        framework_name = self.info["framework_name"]
        if hasattr(self.app_info, "repo"):
            repo_link = self.app_info.repo
            repo_name = repo_link.split("/")[-1]
            return {"url": repo_link, "name": repo_name}

        # @TODO: for unreleased apps
        repo_name = framework_name.replace(" ", "-")
        repo_link = f"https://github.com/supervisely-ecosystem/{repo_name}"
        return {"url": repo_link, "name": repo_name}

    # @TODO: method not used (might be helpful for unreleased apps)
    def _find_app_config(self):
        """
        Find app config.json in project structure.

        :returns: Config dictionary or None if not found
        :rtype: Optional[Dict[str, Any]]
        """
        try:
            current_dir = Path(os.path.abspath(os.path.dirname(__file__)))
            root_dir = current_dir

            while root_dir.parent != root_dir:
                config_path = root_dir / "supervisely_integration" / "train" / "config.json"
                if config_path.exists():
                    break
                root_dir = root_dir.parent

            config_path = root_dir / "supervisely_integration" / "train" / "config.json"
            if config_path.exists():
                return sly_json.load_json_file(config_path)

        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
            return None

    def _get_sample_predictions_gallery(self):
        evaluation_report_id = self.info.get("evaluation_report_id")
        if evaluation_report_id is None:
            return None
        benchmark_file_info = self.api.file.get_info_by_id(evaluation_report_id)
        evaluation_report_path = os.path.dirname(benchmark_file_info.path)
        if os.path.basename(evaluation_report_path) != "visualizations":
            logger.debug(
                f"Visualizations directory is not found in the report directory: '{evaluation_report_path}'"
            )
            return None

        evaluation_report_data_path = os.path.join(evaluation_report_path, "data")

        seek_file = "explore_predictions_gallery_widget"
        remote_gallery_widget_json = None
        for file in self.api.file.list(
            self.team_id, evaluation_report_data_path, False, "fileinfo"
        ):
            if file.name.startswith(seek_file) and file.name.endswith(".json"):
                remote_gallery_widget_json = file.path
                break

        if remote_gallery_widget_json is None:
            logger.debug(
                f"Gallery widget is not found in the report directory: '{evaluation_report_path}'"
            )
            return None

        save_path = os.path.join(
            self.output_dir, "data", "explore_predictions_gallery_widget_expmt.json"
        )
        self.api.file.download(self.team_id, remote_gallery_widget_json, save_path)

        widget_html = """
<sly-iw-gallery ref="gallery_widget_expmt" iw-widget-id="gallery_widget_expmt" :options="{'isModalWindow': false}"
        :actions="{
        'init': {
            'dataSource': '/data/explore_predictions_gallery_widget_expmt.json',
        },
        
    }" :command="command" :data="data">        
    </sly-iw-gallery>
    """
        return widget_html

    def _get_demo_scripts(self):
        """Get demo scripts.

        :returns: Demo scripts
        :rtype: Tuple[dict, dict, dict]
        """

        demo_pytorch_filename = "demo_pytorch.py"
        demo_onnx_filename = "demo_onnx.py"
        demo_tensorrt_filename = "demo_tensorrt.py"

        pytorch_demo = {"path": None, "script": None}
        onnx_demo = {"path": None, "script": None}
        trt_demo = {"path": None, "script": None}

        demo_path = self.app_options.get("demo", {}).get("path")
        if demo_path is None:
            logger.debug("Demo path is not found in app options")
            return pytorch_demo, onnx_demo, trt_demo

        local_demo_dir = os.path.join(os.getcwd(), demo_path)
        if not sly_fs.dir_exists(local_demo_dir):
            logger.debug(f"Demo directory '{local_demo_dir}' does not exist")
            return pytorch_demo, onnx_demo, trt_demo

        local_files = sly_fs.list_files(local_demo_dir)
        for file in local_files:
            if file.endswith(demo_pytorch_filename):
                with open(file, "r", encoding="utf-8") as f:
                    script = f.read()
                pytorch_demo = {"path": file, "script": script}
            elif file.endswith(demo_onnx_filename):
                with open(file, "r", encoding="utf-8") as f:
                    script = f.read()
                onnx_demo = {"path": file, "script": script}
            elif file.endswith(demo_tensorrt_filename):
                with open(file, "r", encoding="utf-8") as f:
                    script = f.read()
                trt_demo = {"path": file, "script": script}

        return pytorch_demo, onnx_demo, trt_demo

    def _get_app_train_serve_app_info(self):
        """Get app slugs.

        :returns: App slugs
        :rtype: Tuple[str, str]
        """

        def find_app_by_framework(api: Api, framework: str, action: Literal["train", "serve"]):
            try:
                modules = api.app.get_list_ecosystem_modules(
                    categories=[action, f"framework:{framework}"],
                    categories_operation="and",
                )
                if len(modules) == 0:
                    return None
                return modules[0]
            except Exception as e:
                logger.warning(f"Failed to find {action} app by framework: {e}")
                return None

        train_app_info = find_app_by_framework(self.api, self.info["framework_name"], "train")
        serve_app_info = find_app_by_framework(self.api, self.info["framework_name"], "serve")

        if train_app_info is not None:
            train_app_slug = train_app_info["slug"].replace("supervisely-ecosystem/", "")
            train_app_id = train_app_info["id"]
        else:
            train_app_slug = None
            train_app_id = None

        if serve_app_info is not None:
            serve_app_slug = serve_app_info["slug"].replace("supervisely-ecosystem/", "")
            serve_app_id = serve_app_info["id"]
        else:
            serve_app_slug = None
            serve_app_id = None

        train_app = {
            "slug": train_app_slug,
            "module_id": train_app_id,
        }
        serve_app = {
            "slug": serve_app_slug,
            "module_id": serve_app_id,
        }
        return train_app, serve_app

    def _get_agent_info(self) -> str:
        task_id = self.info.get("task_id", None)

        agent_info = {
            "name": None,
            "id": None,
            "link": None,
        }

        if task_id is None or task_id == -1:
            return agent_info

        task_info = self.api.task.get_info_by_id(task_id)
        if task_info is not None:
            agent_info["name"] = task_info["agentName"]
            agent_info["id"] = task_info["agentId"]
            agent_info["link"] = f"{self.api.server_address}/nodes/{agent_info['id']}/info"
        return agent_info

    def _get_class_names(self, model_classes: list) -> dict:
        """Get class names from model meta.

        :returns: List of class names
        :rtype: list
        """

        return {
            "string": ", ".join(model_classes),
            "short_string": (
                ", ".join(model_classes[:5] + ["..."])
                if len(model_classes) > 5
                else ", ".join(model_classes)
            ),
            "list": model_classes,
            "short_list": (
                model_classes[:3] + ["..."] if len(model_classes) > 3 else model_classes
            ),
        }

    def _get_predict_app_info(self):
        """
        Get predict app info.

        :returns: Predict app info
        :rtype: dict
        """
        predict_app_slug = "supervisely-ecosystem/apply-nn"
        predict_app_module_id = self.api.app.get_ecosystem_module_id(predict_app_slug)
        predict_app = {"slug": predict_app_slug, "module_id": predict_app_module_id}
        return predict_app

    def _get_app_apply_nn_app_info(self):
        """
        Get apply NN app info.

        :returns: Apply NN app info
        :rtype: dict
        """
        # Images
        apply_nn_images_slug = "nn-image-labeling/project-dataset"
        apply_nn_images_module_id = self.api.app.get_ecosystem_module_id(
            f"supervisely-ecosystem/{apply_nn_images_slug}"
        )
        apply_nn_images_app = {"slug": apply_nn_images_slug, "module_id": apply_nn_images_module_id}

        # Videos
        apply_nn_videos_slug = "apply-nn-to-videos-project"
        apply_nn_videos_module_id = self.api.app.get_ecosystem_module_id(
            f"supervisely-ecosystem/{apply_nn_videos_slug}"
        )
        apply_nn_videos_app = {"slug": apply_nn_videos_slug, "module_id": apply_nn_videos_module_id}
        return apply_nn_images_app, apply_nn_videos_app

    def _get_project_splits(self):
        train_collection_id = self.info.get("train_collection_id", None)
        if train_collection_id is not None:
            train_collection_info = self.api.entities_collection.get_info_by_id(train_collection_id)
            train_collection_name = train_collection_info.name
            train_collection_url = self._datasets_url_with_entities_filter(self.info["project_id"], [{"type": "entities_collection", "data": {"collectionId": train_collection_id, "include": True}}])
            train_size = self.info.get("train_size", "N/A")
        else:
            train_collection_name = None
            train_size = None
            train_collection_url = None

        val_collection_id = self.info.get("val_collection_id", None)
        if val_collection_id is not None:
            val_collection_info = self.api.entities_collection.get_info_by_id(val_collection_id)
            val_collection_name = val_collection_info.name
            val_collection_url = self._datasets_url_with_entities_filter(self.info["project_id"], [{"type": "entities_collection", "data": {"collectionId": val_collection_id, "include": True}}])
            val_size = self.info.get("val_size", "N/A")
        else:
            val_collection_name = None
            val_size = None
            val_collection_url = None

        splits = {
            "train": {
                "name": train_collection_name,
                "size": train_size,
                "url": train_collection_url,
            },
            "val": {
                "name": val_collection_name,
                "size": val_size,
                "url": val_collection_url,
            },
        }
        return splits

    def _get_project_version(self, project_info: ProjectInfo):
        project_version = project_info.version
        if project_version is None:
            version_info = {
                "version": "N/A",
                "id": None,
                "url": None,
            }
        else:
            version_info = {
                "version": project_version["version"],
                "id": project_version["id"],
                "url": f"{self.api.server_address}/projects/{project_info.id}/versions",
            }
        return version_info

    def _get_project_context(self):
        project_id = self.info["project_id"]
        project_info = self.api.project.get_info_by_id(project_id)
        project_version = self._get_project_version(project_info)
        project_type = project_info.type
        project_url = f"{self.api.server_address}/projects/{project_id}/datasets"
        model_classes = [cls.name for cls in self.model_meta.obj_classes]
        class_names = self._get_class_names(model_classes)
        splits = self._get_project_splits()

        project_context = {
            "id": project_id,
            "workspace_id": project_info.workspace_id if project_info else None,
            "name": project_info.name if project_info else "Project was archived",
            "version": project_version,
            "url": project_url if project_info else None,
            "type": project_type if project_info else None,
            "splits": splits,
            "classes": {
                "count": len(model_classes),
                "names": class_names,
            },
        }
        return project_context

    def _get_base_checkpoint_info(self):
        base_checkpoint_name = self.info.get("base_checkpoint", "N/A")
        base_checkpoint_link = self.info.get("base_checkpoint_link", None)
        base_checkpoint_path = None
        if base_checkpoint_link is not None:
            if base_checkpoint_link.startswith("/experiments/"):
                base_checkpoint_info = self.api.file.get_info_by_path(
                    self.team_id, base_checkpoint_link
                )
                base_checkpoint_name = base_checkpoint_info.name
                base_checkpoint_link = base_checkpoint_info.full_storage_url
                base_checkpoint_path = (
                    f"{self.api.server_address}/files/?path={base_checkpoint_info.path}"
                )

        base_checkpoint = {
            "name": base_checkpoint_name,
            "url": base_checkpoint_link,
            "path": base_checkpoint_path,
        }
        return base_checkpoint

    def _get_model_context(self):
        """Return model description part of context."""
        return {
            "name": self.info["model_name"],
            "framework": self.info["framework_name"],
            "base_checkpoint": self._get_base_checkpoint_info(),
            "task_type": self.info["task_type"],
        }

    def _get_training_context(self):
        """Return training-related context (checkpoints, metrics, etc.)."""

        device = self.info.get("device", "N/A")
        training_session = self._get_training_session()
        training_duration = self._get_training_duration()
        hyperparameters = self._generate_hyperparameters_yaml()

        best_checkpoint = self._get_best_checkpoint()
        onnx_checkpoint, trt_checkpoint = self._get_optimized_checkpoints()

        logs_path = self.info.get("logs", {}).get("link")
        logs_url = f"{self.api.server_address}/files/?path={logs_path}" if logs_path else None

        primary_metric = self._get_primary_metric()
        display_metrics = self._get_display_metrics(self.info["task_type"])

        return {
            "device": device,
            "session": training_session,
            "duration": training_duration,
            "hyperparameters": hyperparameters,
            "checkpoints": {
                "pytorch": best_checkpoint,
                "onnx": onnx_checkpoint,
                "tensorrt": trt_checkpoint,
            },
            "export": self.info.get("export"),
            "logs": {"path": logs_path, "url": logs_url},
            "evaluation": {
                "id": self.info.get("evaluation_report_id"),
                "url": self.info.get("evaluation_report_link"),
                "primary_metric": primary_metric,
                "display_metrics": display_metrics,
                "metrics": self.info.get("evaluation_metrics"),
            },
        }

    def _get_experiment_context(self):
        task_id = self.info["task_id"]
        exp_name = self.info["experiment_name"]
        agent_info = self._get_agent_info()
        date = self._get_date()
        project_context = self._get_project_context()
        model_context = self._get_model_context()
        training_context = self._get_training_context()
        artifacts_dir = self.info["artifacts_dir"].rstrip("/")
        experiment_dir = os.path.basename(artifacts_dir)
        checkpoints_dir = os.path.join(artifacts_dir, "checkpoints")

        experiment_context = {
            "task_id": task_id,
            "name": exp_name,
            "agent": agent_info,
            "date": date,
            "project": project_context,
            "model": model_context,
            "training": training_context,
            "paths": {
                "experiment_dir": {
                    "path": experiment_dir,
                    "url": f"{self.api.server_address}/files/?path={experiment_dir.rstrip('/') + '/'}",
                },
                "artifacts_dir": {
                    "path": artifacts_dir,
                    "url": f"{self.api.server_address}/files/?path={artifacts_dir.rstrip('/') + '/'}",
                },
                "checkpoints_dir": {
                    "path": checkpoints_dir,
                    "url": f"{self.api.server_address}/files/?path={checkpoints_dir.rstrip('/') + '/'}",
                },
            },
        }
        return experiment_context

    def _generate_training_plots(self) -> Optional[str]:
        # pip install tbparse plotly kaleido
        if SummaryReader is None or px is None:
            logger.warning("tbparse or plotly is not installed – skipping training plots generation")
            return None

        logs_path = self.info.get("logs", {}).get("link")
        if logs_path is None:
            return None

        events_files: List[str] = []
        remote_log_files = self.api.file.list(self.team_id, logs_path, return_type="fileinfo")
        try:
            for f in remote_log_files:
                if f.name.startswith("events.out.tfevents"):
                    events_files.append(f.path)
        except Exception as e:
            logger.warning(f"Failed to get training logs: {e}")
            return None

        if len(events_files) == 0:
            return None

        tmp_logs_dir = os.path.join(self.output_dir, "logs_tmp")
        sly_fs.mkdir(tmp_logs_dir, True)
        local_event_path = os.path.join(tmp_logs_dir, os.path.basename(events_files[0]))
        try:
            self.api.file.download(self.team_id, events_files[0], local_event_path)
        except Exception as e:
            logger.warning(f"Failed to download training log: {e}")
            return None

        try:
            reader = SummaryReader(local_event_path)
            scalars_df = reader.scalars
        except Exception as e:
            logger.warning(f"Failed to read training log: {e}")
            return None

        if scalars_df is None or scalars_df.empty:
            return None

        tags_to_plot = scalars_df["tag"].unique().tolist()[:12]
        df_plot = scalars_df[scalars_df["tag"].isin(tags_to_plot)]

        try:
            data_dir = os.path.join(self.output_dir, "data")
            if not sly_fs.dir_exists(data_dir):
                sly_fs.mkdir(data_dir, True)

            n_tags = len(tags_to_plot)
            side = min(4, max(2, math.ceil(math.sqrt(n_tags))))
            cols = side
            rows = math.ceil(n_tags / cols)
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=tags_to_plot)

            for idx, tag in enumerate(tags_to_plot, start=1):
                tag_df = df_plot[df_plot["tag"] == tag]
                if tag_df.empty:
                    continue
                row = (idx - 1) // cols + 1
                col = (idx - 1) % cols + 1
                fig.add_trace(
                    go.Scatter(
                        x=tag_df["step"],
                        y=tag_df["value"],
                        mode="lines",
                        name=tag,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

                if tag.startswith("lr"):
                    fig.update_yaxes(tickformat=".0e", row=row, col=col)

            fig.update_layout(
                height=300 * rows,
                width=400 * cols,
                showlegend=False,
            )

            local_img_path = os.path.join(data_dir, "training_plots_grid.png")
            fig.write_image(local_img_path, engine="kaleido")
            sly_fs.remove_dir(tmp_logs_dir)

            img_widget = f"<sly-iw-image src=\"/data/training_plots_grid.png\" :template-base-path=\"templateBasePath\" :options=\"{{ style: {{ width: '70%', height: 'auto' }} }}\" />"
            return img_widget
        except Exception as e:
            logger.warning(f"Failed to build or save static training plot: {e}")
            return None


# Pylint Errors: ************* Module supervisely.api.app_api
# supervisely/api/app_api.py:1463:20: E0606: Possibly using variable 'progress' before assignment (possibly-used-before-assignment)
