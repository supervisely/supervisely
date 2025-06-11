from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
from supervisely import logger
from supervisely.api.api import Api
from supervisely.api.file_api import FileInfo
from supervisely.nn.inference import Inference
from supervisely.nn.task_type import TaskType
from supervisely.nn.utils import RuntimeType
from supervisely.project import ProjectMeta
from supervisely.template.base_generator import BaseGenerator


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
        exp_name = self.info.get("experiment_name", "N/A")
        model_name = self.info.get("model_name", "N/A")
        task_type = self.info.get("task_type", "N/A")
        framework_name = self.info.get("framework_name", "N/A")
        device = self.info.get("device", "N/A")

        project_id = self.info["project_id"]
        project_info = self.api.project.get_info_by_id(project_id)
        project_type = project_info.type
        project_link = f"{self.api.server_address}/projects/{project_id}/datasets"
        project_train_size = self.info["train_size"]
        project_val_size = self.info["val_size"]
        model_classes = [cls.name for cls in self.model_meta.obj_classes]
        class_names = self._get_class_names(model_classes)

        date = self._get_date()
        training_duration = self.info.get("training_duration", "N/A")
        metrics = self._generate_metrics_table()
        primary_metric = self._get_primary_metric()
        display_metrics = self._get_display_metrics(task_type)
        checkpoints = self._generate_checkpoints_table()
        hyperparameters = self._generate_hyperparameters_yaml()
        artifacts_dir = self.info["artifacts_dir"].rstrip("/")
        experiment_dir = os.path.basename(artifacts_dir)
        docker_image = self._get_docker_image()
        repo_info = self._get_repository_info()
        training_session = self._get_training_session()

        best_checkpoint = self._get_best_checkpoint()
        onnx_checkpoint, trt_checkpoint = self._get_optimized_checkpoints()
        sample_pred_gallery = self._get_sample_predictions_gallery()
        pytorch_demo, onnx_demo, trt_demo = self._get_demo_scripts()
        train_app_slug, serve_app_slug = self._get_app_slugs()
        agent_info = self._get_agent_info()

        context = {
            "env": {
                "server_address": self.api.server_address,
            },
            "experiment": {
                "name": exp_name,
                "model_name": model_name,
                "task_name": task_type,
                "device": device,
                "framework_name": framework_name,
                "date": date,
                "training_duration": training_duration,
                "artifacts_dir": artifacts_dir,
                "export": self.info.get("export"),
                "agent": agent_info,
            },
            "project": {
                "name": project_info.name if project_info else "",
                "link": project_link,
                "type": project_type,
                "train_size": project_train_size,
                "val_size": project_val_size,
                "classes_count": len(model_classes),
                "class_names": class_names,
            },
            "links": {
                "app": {
                    "train": train_app_slug,
                    "serve": serve_app_slug,
                    "apply_nn_to_images": "nn-image-labeling/project-dataset",
                    "apply_nn_to_videos": "apply-nn-to-videos-project",
                },
                "training_session": training_session,
                "evaluation_report": {
                    "id": self.info.get("evaluation_report_id"),
                    "url": self.info.get("evaluation_report_link"),
                },
                "tensorboard_logs": {
                    "path": self.info.get("logs", {}).get("link", None),
                    "url": (
                        f"{self.api.server_address}/files/?path={self.info.get('logs', {}).get('link')}"
                        if self.info.get("logs", {}).get("link")
                        else None
                    ),
                },
                "team_files": {
                    "path": artifacts_dir,
                    "url": (
                        f"{self.api.server_address}/files/?path={self.artifacts_dir}"
                        if self.artifacts_dir
                        else None
                    ),
                },
                "checkpoint_dir_url": f"{self.api.server_address}/files/?path={self.artifacts_dir}",
            },
            "artifacts": {
                "checkpoints_table": checkpoints,
                "metrics_table": metrics,
                "hyperparameters": hyperparameters,
                "experiment_dir": experiment_dir,
                "best_checkpoint": best_checkpoint,
                "onnx_checkpoint": onnx_checkpoint,
                "trt_checkpoint": trt_checkpoint,
            },
            "benchmark": {
                "id": self.info.get("evaluation_report_id"),
                "url": self.info.get("evaluation_report_link"),
                "metrics": self.info.get("evaluation_metrics"),
                "primary_metric": primary_metric,
                "display_metrics": display_metrics,
            },
            "code": {
                "docker": {
                    "image": docker_image,
                },
                "local_prediction": {
                    "repo": {
                        "name": repo_info["name"],
                        "url": repo_info["url"],
                    },
                    "serving_module": self.serving_class.__module__ if self.serving_class else None,
                    "serving_class": self.serving_class.__name__ if self.serving_class else None,
                },
                "demo": {
                    "pytorch": pytorch_demo,
                    "onnx": onnx_demo,
                    "tensorrt": trt_demo,
                },
            },
            "widgets": {
                "sample_pred_gallery": sample_pred_gallery,
            },
        }

        return context

    def _generate_metrics_table(self) -> str:
        """Generate HTML table with evaluation metrics.

        :returns: HTML string with metrics table
        :rtype: str
        """
        metrics = self.info.get("evaluation_metrics", {})
        if not metrics:
            return None

        html = ['<table class="metrics-table">']
        html.append("<thead><tr><th>Metrics</th><th>Values</th></tr></thead>")
        html.append("<tbody>")

        for metric_name, metric_value in metrics.items():
            metric_name = metric_name.replace("_", " ")
            metric_name = metric_name.replace("-", " ")
            if isinstance(metric_value, float):
                metric_value = f"{metric_value:.4f}"
            html.append(f"<tr><td>{metric_name}</td><td>{metric_value}</td></tr>")

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

        html = ['<table class="metrics-table">']

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
                    {"name": metric_name, "value": value, "percent_value": percent_value}
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

        html = ['<table class="checkpoints-table">']
        html.append("<thead><tr><th>Checkpoints</th><th>Size</th><th> </th></tr></thead>")
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
        task_id = self.info.get("task_id", None)
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

    def _get_app_slugs(self):
        """Get app slugs.

        :returns: App slugs
        :rtype: Tuple[str, str]
        """

        def find_app_by_framework(api: Api, framework: str, action: Literal["train", "serve"]):
            modules = api.app.get_list_ecosystem_modules(
                categories=[action, f"framework:{framework}"], categories_operation="and"
            )
            if len(modules) == 0:
                return None
            return modules[0]

        train_app_info = find_app_by_framework(self.api, self.info["framework_name"], "train")
        serve_app_info = find_app_by_framework(self.api, self.info["framework_name"], "serve")

        train_app_slug = train_app_info["slug"].replace("supervisely-ecosystem/", "")
        serve_app_slug = serve_app_info["slug"].replace("supervisely-ecosystem/", "")

        return train_app_slug, serve_app_slug

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
