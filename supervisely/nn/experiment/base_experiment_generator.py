import json
import os
import shutil
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from jinja2 import Environment, FileSystemLoader, Template

import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
from supervisely import logger
from supervisely.api.api import Api
from supervisely.nn.inference import Inference
from supervisely.nn.utils import RuntimeType
from supervisely.project import ProjectMeta


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

        self.artifacts_dir = self.info["artifacts_dir"]
        self.serving_class = serving_class
        self.app_info = self._get_app_info()

        self.report_name = "Experiment Report.lnk"
        self.jinja_env = Environment(
            loader=FileSystemLoader(Path(__file__).parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            # pip install jinja-markdown
            extensions=["jinja_markdown.MarkdownExtension"],
        )

        self.output_dir = "./experiment_report"
        sly_fs.mkdir(self.output_dir, True)

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

        classes = [cls.name for cls in self.model_meta.obj_classes]
        date = self._get_date()
        links = self._generate_links()
        metrics = self._generate_metrics_table()
        checkpoints = self._generate_checkpoints_table()
        hyperparameters = self._generate_hyperparameters_yaml()
        experiment_dir = os.path.basename(self.info["artifacts_dir"])
        docker_image = self._get_docker_image()
        repo_info = self._get_repository_info()
        buttons = self._get_buttons()

        # Generate code blocks
        model_api_code = self._generate_model_api_code(experiment_dir)
        docker_code = self._generate_docker_code(docker_image, experiment_dir)
        local_prediction_code = (
            self._generate_local_prediction_code(
                repo_info["url"], repo_info["name"], experiment_dir, self.info["best_checkpoint"]
            )
            if self.serving_class
            else None
        )

        optimized_model_code = (
            self._generate_optimized_model_code(repo_info["name"], experiment_dir)
            if self.info.get("export")
            else None
        )

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
            "model_api_code": model_api_code,
            "docker_code": docker_code,
            "local_prediction_code": local_prediction_code,
            "optimized_model_code": optimized_model_code,
            "has_onnx": self.info.get("export")
            and any("onnx" in k.lower() for k in self.info["export"].keys()),
            "has_tensorrt": self.info.get("export")
            and any("engine" in k.lower() for k in self.info["export"].keys()),
        }

    def _generate_model_api_code(self, experiment_dir: str) -> str:
        """Generate code for Model API section."""
        return """import supervisely as sly

api = sly.Api()

# Deploy
model = api.nn.deploy_custom_model(
    checkpoint_id={checkpoint_path},  # file id
)

# Predict
prediction = model.predict(
    images="image.png"  # image | path | url
)"""

    def _generate_docker_code(self, docker_image: str, experiment_dir: str) -> str:
        """Generate code blocks for Docker section."""
        docker_pull = f"docker pull {docker_image}"

        docker_run = f"""docker run \\
  --runtime=nvidia \\
  -v "./{experiment_dir}:/model" \\
  -p 8000:8000 \\
  {docker_image} \\
  predict \\
  "./image.jpg" \\
  --model "/model" \\
  --device "cuda:0\""""

        return {"docker_pull": docker_pull, "docker_run": docker_run}

    def _generate_local_prediction_code(
        self, repo_url: str, repo_name: str, experiment_dir: str, best_checkpoint: str
    ) -> str:
        """Generate code blocks for Local Prediction section."""
        git_clone = f"""git clone {repo_url}
cd {repo_name}"""

        install_requirements = """pip install -r dev_requirements.txt
pip install supervisely"""

        inference_code = f"""# Be sure you are in the root of the {repo_name} repository
from {self.serving_class.__module__} import {self.serving_class.__name__}

# Load model
model = {self.serving_class.__name__}(
    checkpoint="./{experiment_dir}/checkpoints/{best_checkpoint}",  # path to the checkpoint
    device="cuda",
)

# Predict
prediction = model(
    "image.png",  # local paths, directory, local project, np.array, PIL.Image, URL
    params={{"confidence_threshold": 0.5}}
)"""

        return {
            "git_clone": git_clone,
            "install_requirements": install_requirements,
            "inference_code": inference_code,
        }

    def _generate_optimized_model_code(self, repo_name: str, experiment_dir: str) -> str:
        """Generate code block for optimized model (ONNX/TensorRT)."""
        has_onnx = self.info.get("export") and any(
            "onnx" in k.lower() for k in self.info["export"].keys()
        )
        model_type = "best.onnx" if has_onnx else "best.engine"

        return f"""# Be sure you are in the root of the {repo_name} repository
from {self.serving_class.__module__} import {self.serving_class.__name__}

model = {self.serving_class.__name__}(
    model_dir="./{experiment_dir}",
    checkpoint="{model_type}",
    device="cuda",
)"""

    def _get_buttons(self) -> str:
        """Returns HTML code for template buttons."""
        export = self.info.get("export", {})
        has_tensorrt = export and any("engine" in k.lower() for k in export.keys())
        has_onnx = export and any("onnx" in k.lower() for k in export.keys())

        html = ['<ul class="buttons-list">']
        html.append('<li><span class="button-icon">🚀</span> Deploy (PyTorch)</li>')

        if has_tensorrt:
            html.append('<li><span class="button-icon">🚀</span> Deploy (TensorRT)</li>')
        elif has_onnx:
            html.append('<li><span class="button-icon">🚀</span> Deploy (ONNX)</li>')

        html.extend(
            [
                '<li><span class="button-icon">⏩</span> Fine-tune</li>',
                '<li><span class="button-icon">🔄</span> Re-train</li>',
                '<li><span class="button-icon">📦</span> Download model</li>',
                '<li><span class="button-icon">❌</span> Remove permamently</li>',
            ]
        )

        html.append("</ul>")
        return "\n".join(html)

    def _generate_links(self) -> str:
        """Generate links to related resources."""
        html = ['<ul class="links-list">']

        task_id = self.info.get("task_id")
        if task_id:
            html.append(
                f'<li><span class="link-icon">🎓</span> <a href="{self.api.server_address}/apps/sessions/{task_id}">Training Task</a></li>'
            )

        eval_id = self.info.get("evaluation_report_id")
        eval_link = self.info.get("evaluation_report_link")
        if eval_id:
            report_link = eval_link or f"{self.api.server_address}/model-benchmark?id={eval_id}"
            html.append(
                f'<li><span class="link-icon">📊</span> <a href="{report_link}">Evaluation Report</a></li>'
            )

        logs = self.info.get("logs", {})
        if logs and "link" in logs:
            html.append(
                f'<li><span class="link-icon">⚡</span> <a href="{self.api.server_address}/files/?path={logs["link"]}">TensorBoard Logs</a></li>'
            )

        artifacts_dir = self.info.get("artifacts_dir")
        if artifacts_dir:
            html.append(
                f'<li><span class="link-icon">💾</span> <a href="{self.api.server_address}/files/?path={artifacts_dir}">Open in Team Files</a></li>'
            )

        html.append("</ul>")
        return "\n".join(html)

    def _generate_metrics_table(self) -> str:
        """Generate metrics table in HTML format."""
        metrics = self.info.get("evaluation_metrics", {})
        if not metrics:
            return ""

        html = ['<table class="metrics-table">']
        html.append("<thead><tr><th>Metrics</th><th>Values</th></tr></thead>")
        html.append("<tbody>")

        primary_metric = self.info.get("primary_metric")

        common_metrics = [
            "mAP",
            "AP50",
            "AP75",
            "f1",
            "precision",
            "recall",
            "accuracy",
            "mean_iou",
            "pixel_accuracy",
        ]

        if primary_metric and primary_metric in metrics:
            metric_value = metrics[primary_metric]
            if isinstance(metric_value, float):
                metric_value = f"{metric_value:.4f}"
            html.append(f"<tr><td>{primary_metric}</td><td>{metric_value}</td></tr>")

        for metric_name in common_metrics:
            if metric_name in metrics and metric_name != primary_metric:
                metric_value = metrics[metric_name]
                if isinstance(metric_value, float):
                    metric_value = f"{metric_value:.4f}"
                html.append(f"<tr><td>{metric_name}</td><td>{metric_value}</td></tr>")

        for metric_name, metric_value in metrics.items():
            if metric_name not in common_metrics and metric_name != primary_metric:
                if isinstance(metric_value, float):
                    metric_value = f"{metric_value:.4f}"
                html.append(f"<tr><td>{metric_name}</td><td>{metric_value}</td></tr>")

        html.append("</tbody>")
        html.append("</table>")
        return "\n".join(html)

    def _generate_checkpoints_table(self) -> str:
        """Generate checkpoints table in HTML format."""
        pytorch_checkpoints = self.info.get("checkpoints", [])
        if not pytorch_checkpoints:
            return ""

        optimized_checkpoints = []
        export = self.info.get("export", {})
        if export:
            onnx_checkpoint = export.get(RuntimeType.ONNX)
            if onnx_checkpoint:
                optimized_checkpoints.append(onnx_checkpoint)
            engine_checkpoint = export.get(RuntimeType.TENSORRT)
            if engine_checkpoint:
                optimized_checkpoints.append(engine_checkpoint)

        checkpoints = pytorch_checkpoints + optimized_checkpoints

        checkpoint_paths = [
            os.path.join(self.artifacts_dir, checkpoint) for checkpoint in checkpoints
        ]
        checkpoint_infos = [
            self.api.file.get_info_by_path(self.team_id, checkpoint_path)
            for checkpoint_path in checkpoint_paths
        ]
        checkpoint_sizes = [f"{info.sizeb / 1024 / 1024:.2f} MB" for info in checkpoint_infos]

        remote_checkpoint_paths = [
            urllib.parse.quote(info.path, safe="/") for info in checkpoint_infos
        ]
        remote_checkpoint_paths = [
            urllib.parse.quote(level1, safe="") for level1 in remote_checkpoint_paths
        ]

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
        """Generate YAML with hyperparameters."""
        hyperparams_yaml = None

        if self.hyperparameters is not None:
            if isinstance(self.hyperparameters, dict):
                hyperparams_yaml = yaml.dump(self.hyperparameters, default_flow_style=False)
            elif isinstance(self.hyperparameters, str):
                if self.hyperparameters.strip().startswith("{"):
                    try:
                        hyperparams_dict = json.loads(self.hyperparameters)
                        hyperparams_yaml = yaml.dump(hyperparams_dict, default_flow_style=False)
                    except Exception:
                        hyperparams_yaml = self.hyperparameters
                else:
                    hyperparams_yaml = self.hyperparameters

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
        template_file = self.api.file.upload(
            team_id=self.team_id, src=template_path, dst=remote_template_path
        )

        # state.json
        state_path = self.generate_state()
        remote_state_path = os.path.join(remote_dir, "state.json")
        state_file = self.api.file.upload(
            team_id=self.team_id, src=state_path, dst=remote_state_path
        )

        # report.lnk
        remote_report_path = os.path.join(remote_dir, self.report_name)
        report_path = self.generate_report_link(template_file.id)
        report_file = self.api.file.upload(
            team_id=self.team_id, src=report_path, dst=remote_report_path
        )
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
