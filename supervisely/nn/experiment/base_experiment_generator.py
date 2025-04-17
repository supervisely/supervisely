import json
from pathlib import Path
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import yaml
from jinja2 import Template, Environment, FileSystemLoader

from supervisely.api.api import Api
from supervisely.nn.inference import Inference
from supervisely.project import ProjectMeta
import supervisely.io.json as sly_json
import supervisely.io.fs as sly_fs
import supervisely.io.env as sly_env
from supervisely import logger


class BaseExperimentGenerator:
    """
    Base class for generating experiment reports.

    Provides functionality for generating experiment report pages in Markdown format.
    """

    def __init__(
        self,
        api: Api,
        experiment_info: dict,
        hyperparameters: str,
        model_meta: ProjectMeta,
        serving_class: Optional[Inference] = None,
    ):
        """Initialize experiment generator class.

        :param experiment_info: Dictionary with experiment information
        :type experiment_info: Dict[str, Any]
        :param hyperparameters: Hyperparameters as YAML string or dictionary
        :type hyperparameters: Optional[Union[str, Dict]]
        :param model_meta: Model metadata as dictionary
        :type model_meta: Optional[Union[str, Dict]]
        :param serving_class: Serving class for model inference
        :type serving_class: Optional[Inference]
        """
        self.api = api
        self.team_id = sly_env.team_id()
        self.info = experiment_info
        self.hyperparameters = hyperparameters
        self.model_meta = model_meta

        self.output_dir = "./experiment_report"
        sly_fs.mkdir(self.output_dir, True)

        self.artifacts_dir = self.info["artifacts_dir"]

        self.serving_class = serving_class
        self.app_info = self._get_app_info()

        self.report_name = "Experiment Report.lnk"
        
        # Настройка окружения Jinja
        self.jinja_env = Environment(
            loader=FileSystemLoader(Path(__file__).parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )

    def get_state(self):
        """Get state for state.json"""
        return {}
    
    def generate_template(self) -> str:
        """Generate experiment report HTML and save it as template.vue.

        :return: Path to the generated template.vue file
        :rtype: str
        """
        context = self._prepare_template_context()
        template = self.jinja_env.get_template("report_template.html.jinja")
        html_content = template.render(**context)
        template_path = os.path.join(self.output_dir, "template.vue")
        with open(template_path, "w") as f:
            f.write(html_content)
        return template_path
    
    def _prepare_template_context(self) -> Dict[str, Any]:
        """Prepare context for Jinja template.
        
        :return: Dictionary with data for template
        :rtype: Dict[str, Any]
        """
        # Experiment
        exp_name = self.info["experiment_name"]
        model_name = self.info["model_name"]
        task_type = self.info["task_type"]
        framework_name = self.info["framework_name"]
        
        # Project
        project_id = self.info["project_id"]
        project_info = self.api.project.get_info_by_id(project_id)
        project_type = project_info.type
        project_link = f"{self.api.server_address}/projects/{project_id}/datasets"
        
        # Train / Val sizes
        train_size = self.info["train_size"]
        val_size = self.info["val_size"]
        
        # Classes
        classes = [cls.name for cls in self.model_meta.obj_classes]
        # Date
        date = self._get_date()
        # Links to related resources
        links = self._generate_links()
        # Metrics
        metrics = self._generate_metrics_table()
        # Checkpoints
        checkpoints = self._generate_checkpoints_table()
        # Hyperparameters
        hyperparameters = self._generate_hyperparameters_yaml()
        # Deployment information
        experiment_dir = os.path.basename(self.info["artifacts_dir"])
        # Docker
        docker_image = self._get_docker_image()
        # Repository information
        repo_info = self._get_repository_info()
        # Buttons
        buttons = self._get_buttons()
        
        return {
            "experiment_name": exp_name,
            "model_name": model_name,
            "task_name": task_type,
            "framework_name": framework_name,
            "project_name": project_info.name if project_info else "",
            "project_link": project_link,
            "train_size": train_size,
            "train_type": project_type,
            "validation_size": val_size,
            "classes_count": len(classes),
            "class_names": ", ".join(classes),
            "date": date,
            "links": links,
            "metrics_table": metrics,
            "checkpoints_table": checkpoints,
            "hyperparameters": hyperparameters,
            "experiment_dir": experiment_dir,
            "docker_image": docker_image,
            "repository_url": repo_info.get("url", ""),
            "repository_name": repo_info.get("name", ""),
            "checkpoint_path": "{checkpoint_id}",
            "checkpoint_dir_url": f"{self.api.server_address}/files/?path={self.artifacts_dir}",
            "buttons": buttons,
            "serving_class": self.serving_class.__name__ if self.serving_class else None,
            "serving_module": self.serving_class.__module__ if self.serving_class else None,
            "best_checkpoint": self.info["best_checkpoint"],
        }
    
    def _get_buttons(self) -> str:
        """Returns HTML code for template buttons."""
        export = self.info.get("export", {})
        has_tensorrt = export and any("engine" in k.lower() for k in export.keys())
        has_onnx = export and any("onnx" in k.lower() for k in export.keys())
        
        buttons = []
        buttons.append("- 🚀 Deploy (PyTorch)")
        
        if has_tensorrt:
            buttons.append("- 🚀 Deploy (TensorRT)")
        elif has_onnx:
            buttons.append("- 🚀 Deploy (ONNX)")
            
        buttons.extend([
            "- ⏩ Fine-tune",
            "- 🔄 Re-train",
            "- 📦 Download model",
            "- ❌ Remove permamently",
        ])
        
        return "\n".join(buttons)
    
    def _generate_links(self) -> str:
        """Генерирует ссылки на связанные ресурсы."""
        lines = []
        
        # Training task link
        task_id = self.info.get("task_id")
        if task_id:
            lines.append(
                f"- 🎓 [Training Task]({self.api.server_address}/apps/sessions/{task_id})"
            )
        
        # Evaluation report link
        eval_id = self.info.get("evaluation_report_id")
        eval_link = self.info.get("evaluation_report_link")
        if eval_id:
            report_link = eval_link or f"{self.api.server_address}/model-benchmark?id={eval_id}"
            lines.append(f"- 📊 [Evaluation Report]({report_link})")
        
        # TensorBoard logs link
        logs = self.info.get("logs", {})
        if logs and "link" in logs:
            lines.append(f"- ⚡ [TensorBoard Logs]({self.api.server_address}/files/?path={logs['link']})")
        
        # Artifacts link in Team Files
        artifacts_dir = self.info.get("artifacts_dir")
        if artifacts_dir:
            lines.append(
                f"- 💾 [Open in Team Files]({self.api.server_address}/files/?path={artifacts_dir})"
            )
        
        return "\n".join(lines)
    
    def _generate_metrics_table(self) -> str:
        """Генерирует таблицу метрик."""
        metrics = self.info.get("evaluation_metrics", {})
        if not metrics:
            return ""
        
        lines = [
            "| Metrics | Values |",
            "|---------|--------|",
        ]
        
        # Get primary metric if specified
        primary_metric = self.info.get("primary_metric")
        
        # List of important metrics by task types
        common_metrics = [
            "mAP", "AP50", "AP75", "f1", "precision", "recall", 
            "accuracy", "mean_iou", "pixel_accuracy",
        ]
        
        # Add primary metric first if specified
        if primary_metric and primary_metric in metrics:
            metric_value = metrics[primary_metric]
            if isinstance(metric_value, float):
                metric_value = f"{metric_value:.4f}"
            lines.append(f"| {primary_metric} | {metric_value} |")
        
        # Add other important metrics
        for metric_name in common_metrics:
            if metric_name in metrics and metric_name != primary_metric:
                metric_value = metrics[metric_name]
                if isinstance(metric_value, float):
                    metric_value = f"{metric_value:.4f}"
                lines.append(f"| {metric_name} | {metric_value} |")
        
        # Add remaining metrics
        for metric_name, metric_value in metrics.items():
            if metric_name not in common_metrics and metric_name != primary_metric:
                if isinstance(metric_value, float):
                    metric_value = f"{metric_value:.4f}"
                lines.append(f"| {metric_name} | {metric_value} |")
        
        return "\n".join(lines)
    
    def _generate_checkpoints_table(self) -> str:
        """Генерирует таблицу чекпоинтов."""
        checkpoints = self.info.get("checkpoints", [])
        if not checkpoints:
            return ""
        
        lines = [
            "| Checkpoints |",
            "|---------|",
        ]
        
        for checkpoint in checkpoints:
            if isinstance(checkpoint, str):
                lines.append(f"| {os.path.basename(checkpoint)} |")
        
        return "\n".join(lines)
    
    def _generate_hyperparameters_yaml(self) -> str:
        """Генерирует YAML с гиперпараметрами."""
        hyperparams_yaml = None
        
        # First try to use directly provided hyperparameters
        if self.hyperparameters is not None:
            if isinstance(self.hyperparameters, dict):
                hyperparams_yaml = yaml.dump(self.hyperparameters, default_flow_style=False)
            elif isinstance(self.hyperparameters, str):
                if self.hyperparameters.strip().startswith("{"):
                    # If it's a JSON string, convert to YAML
                    try:
                        hyperparams_dict = json.loads(self.hyperparameters)
                        hyperparams_yaml = yaml.dump(hyperparams_dict, default_flow_style=False)
                    except Exception:
                        hyperparams_yaml = self.hyperparameters
                else:
                    # Assume it's already YAML
                    hyperparams_yaml = self.hyperparameters
        
        # If not found, try to load from downloaded file
        if hyperparams_yaml is None and "hyperparameters" in self.info and self.artifacts_dir:
            hyperparams_path = os.path.join(
                self.artifacts_dir, os.path.basename(self.info["hyperparameters"])
            )
            
            if os.path.exists(hyperparams_path):
                try:
                    with open(hyperparams_path, "r") as f:
                        if hyperparams_path.endswith(".yaml") or hyperparams_path.endswith(".yml"):
                            hyperparams = yaml.safe_load(f)
                            hyperparams_yaml = yaml.dump(hyperparams, default_flow_style=False)
                        else:
                            hyperparams_yaml = f.read()
                except Exception as e:
                    print(f"Failed to read hyperparameters: {e}")
        
        return hyperparams_yaml or ""
    
    def _get_docker_image(self) -> str:
        """Returns Docker image."""
        framework_name = self.info.get("framework_name")
        if not framework_name:
            return "supervisely/nvidia:latest"
        
        if self.app_info:
            docker_image = self.app_info.config.get("docker_image")
            if docker_image:
                return docker_image
                
        return f"supervisely/{framework_name.lower()}:latest"
    
    def _get_repository_info(self) -> Dict[str, str]:
        """Returns repository info."""
        framework_name = self.info.get("framework_name", "")
        
        if self.app_info and hasattr(self.app_info, "repo"):
            repo_link = self.app_info.repo
            repo_name = repo_link.split("/")[-1]
            return {"url": repo_link, "name": repo_name}
            
        # Fallback
        repo_link = f"https://github.com/supervisely-ecosystem/{framework_name.replace(' ', '-')}"
        repo_name = framework_name.replace(" ", "-")
        return {"url": repo_link, "name": repo_name}
    
    def generate_state(self):
        """Generate state for state.json"""
        state = self.get_state()
        state_path = os.path.join(self.output_dir, "state.json")
        sly_json.dump_json_file(state, state_path)
        return state_path
    
    def generate_report_link(self, template_id: int):
        """Generate report link"""
        report_path = os.path.join(self.output_dir, self.report_name)
        with open(report_path, "w") as f:
            f.write(self._get_report_link(template_id))
        return report_path

    def generate_report(self) -> str:
        """Generate and upload report to Supervisely"""
        remote_dir = os.path.join(self.info["artifacts_dir"], "visualization")
        
        # template.vue
        template_path = self.generate_template()
        remote_template_path = os.path.join(remote_dir, "template.vue")
        template_file = self.api.file.upload(team_id=self.team_id, src=template_path, dst=remote_template_path)
        
        # state.json
        state_path = self.generate_state()
        remote_state_path = os.path.join(remote_dir, "state.json")
        state_file = self.api.file.upload(team_id=self.team_id, src=state_path, dst=remote_state_path)
        
        # report.lnk
        remote_report_path = os.path.join(remote_dir, self.report_name)
        report_path = self.generate_report_link(template_file.id)
        report_file = self.api.file.upload(team_id=self.team_id, src=report_path, dst=remote_report_path)
        logger.info("Experiment report generated successfully")
        
        return remote_dir

    def _get_report_link(self, template_id: int):
        """Get path to report file."""
        return self.api.server_address + "/nn/experiments/" + str(template_id)

    def _get_app_info(self):
        try:
            task_info = self.api.task.get_info_by_id(self.info.get("task_id"))
            app_id = task_info["meta"]["app"]["id"]
            app_info = self.api.app.get_info_by_id(app_id)
            return app_info
        except Exception as e:
            print(f"Failed to load app config: {e}")
            return None

    def _find_app_config(self):
        """Find app config in the project"""
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
            print(f"Failed to load config.json: {e}")

    def _get_date(self):
        """Get date"""
        date_str = self.info.get("datetime", "")
        date = date_str
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                date = dt.strftime("%d %b %Y")
            except ValueError:
                pass
        return date