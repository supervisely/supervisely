from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
from supervisely import logger
from supervisely.api.api import Api
from supervisely.nn.inference import Inference
from supervisely.nn.utils import RuntimeType
from supervisely.project import ProjectMeta
from supervisely.template.base_generator import BaseGenerator


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

    def _report_url(self, server_address: str, template_id: int) -> str:
        return f"{server_address}/nn/experiments/{template_id}"

    def upload_to_artifacts(self):
        remote_dir = os.path.join(self.info["artifacts_dir"], "visualization")
        self.upload(remote_dir, team_id=self.team_id)

    def state(self) -> dict:
        return {}
    
    def context(self) -> dict:
        exp_name = self.info["experiment_name"]
        model_name = self.info["model_name"]
        task_type = self.info["task_type"]
        framework_name = self.info["framework_name"]

        project_id = self.info["project_id"]
        project_info = self.api.project.get_info_by_id(project_id)
        project_type = project_info.type
        project_link = f"{self.api.server_address}/projects/{project_id}/datasets"
        project_train_size = self.info["train_size"]
        project_val_size = self.info["val_size"]
        model_classes = [cls.name for cls in self.model_meta.obj_classes]
        
        date = self._get_date()
        metrics = self._generate_metrics_table()
        checkpoints = self._generate_checkpoints_table()
        hyperparameters = self._generate_hyperparameters_yaml()
        artifacts_dir = self.info["artifacts_dir"].rstrip("/")
        experiment_dir = os.path.basename(artifacts_dir)
        docker_image = self._get_docker_image()
        repo_info = self._get_repository_info()
        has_onnx, has_tensorrt, optimized_checkpoint = self._get_optimized_checkpoint()

        context = {
            "experiment": {
                "name": exp_name,
                "model_name": model_name,
                "task_name": task_type,
                "framework_name": framework_name,
                "date": date,
                "artifacts_dir": artifacts_dir,
                "export": self.info.get("export"),
            },
            "project": {
                "name": project_info.name if project_info else "",
                "link": project_link,
                "type": project_type,
                "train_size": project_train_size,
                "val_size": project_val_size,
                "classes_count": len(model_classes),
                "class_names": ", ".join(model_classes),
            },
            "links": {
                "training_session": {
                    "id": self.info.get("task_id"),
                    "url": f"{self.api.server_address}/apps/sessions/{self.info.get('task_id')}" if self.info.get("task_id") else None,
                },
                "evaluation_report": {
                    "id": self.info.get("evaluation_report_id"),
                    "url": self.info.get("evaluation_report_link"),
                },
                "tensorboard_logs": {
                    "path": self.info.get("logs", {}).get("link", None),
                    "url": f"{self.api.server_address}/files/?path={self.info.get('logs', {}).get('link')}" if self.info.get("logs", {}).get("link") else None,
                },
                "team_files": {
                    "path": artifacts_dir,
                    "url": f"{self.api.server_address}/files/?path={self.artifacts_dir}" if self.artifacts_dir else None,
                },
                "checkpoint_dir_url": f"{self.api.server_address}/files/?path={self.artifacts_dir}",
            },
            "artifacts": {
                "checkpoints_table": checkpoints,
                "metrics_table": metrics,
                "hyperparameters": hyperparameters,
                "experiment_dir": experiment_dir,
                "best_checkpoint": self.info["best_checkpoint"],
                "optimized_checkpoint": optimized_checkpoint,
            },
            "code": {
                "docker": {
                    "image": docker_image,
                },
                "local_prediction": {
                    "repo_url": repo_info["url"],
                    "repo_name": repo_info["name"],
                    "serving_module": self.serving_class.__module__ if self.serving_class else None,
                    "serving_class": self.serving_class.__name__ if self.serving_class else None,
                },
            },
            "buttons": {
                "has_onnx": has_onnx,
                "has_tensorrt": has_tensorrt,
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
            if isinstance(metric_value, float):
                metric_value = f"{metric_value:.4f}"
            html.append(f"<tr><td>{metric_name}</td><td>{metric_value}</td></tr>")

        html.append("</tbody>")
        html.append("</table>")
        return "\n".join(html)

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
            for runtime_type in [RuntimeType.ONNXRUNTIME, RuntimeType.TENSORRT]:
                if checkpoint := export.get(runtime_type):
                    checkpoints.append(checkpoint)

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
        html.append("<thead><tr><th>Checkpoints</th><th>Size</th><th>Download</th></tr></thead>")
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
        task_id = self.info["task_id"]
        task_info = self.api.task.get_info_by_id(task_id)
        app_id = task_info["meta"]["app"]["id"]
        return self.api.app.get_info_by_id(app_id)

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

    def _get_optimized_checkpoint(self) -> Tuple[bool, bool, Optional[str]]:
        """Get optimized checkpoint filename (ONNX or TensorRT).
        
        :returns: Checkpoint filename or None if not available
        :rtype: Optional[str]
        """
        export = self.info.get("export", {})
        has_onnx, has_tensorrt, optimized_checkpoint = False, False, None
        if export:
            if optimized_checkpoint := export.get(RuntimeType.TENSORRT):
                has_tensorrt = True
            if optimized_checkpoint := export.get(RuntimeType.ONNXRUNTIME):
                has_onnx = True
        return has_onnx, has_tensorrt, optimized_checkpoint

    def _get_docker_image(self) -> str:
        """Get Docker image for model.
        
        :returns: Docker image name
        :rtype: str
        """
        docker_image = self.app_info.config["docker_image"]
        if not docker_image:
            raise ValueError("Docker image is not found in app config")
        return docker_image

    def _get_repository_info(self) -> Dict[str, str]:
        """Get repository information.
        
        :returns: Dictionary with repo URL and name
        :rtype: Dict[str, str]
        """
        framework_name = self.info["framework_name"]
        if self.app_info and hasattr(self.app_info, "repo"):
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
        
# @TODO: Partly supports unreleased apps
