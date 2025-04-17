import json
from pathlib import Path
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional, Union

import yaml
from jinja2 import Template

from supervisely.api.api import Api
from supervisely.nn.inference import Inference
from supervisely.project import ProjectMeta
import supervisely.io.json as sly_json
import supervisely.io.fs as sly_fs
import supervisely.io.env as sly_env
from supervisely.task.progress import tqdm_sly
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
        self.main_template = self._load_report_template()

    def get_state(self):
        """Get state for state.json"""
        return {}
    
    def _load_report_template(self) -> str:
        """Loads report template from file or returns built-in template."""
        template_path = Path(__file__).parent / "report_template.html"
        return template_path.read_text()
  
    def generate_template(self) -> str:
        """Generate experiment report HTML and save it as template.vue.

        :return: Path to the generated template.vue file
        :rtype: str
        """
        content = self._generate_md()
        content_escaped = content.replace('`', r'\`').replace('"', r'\"').replace('\n', r'\n')
        
        # Используем двойные кавычки снаружи, одинарные внутри для лучшей совместимости
        markdown_widget = f'<sly-markdown-widget content="{content_escaped}"></sly-markdown-widget>'
        
        # Убедимся, что шаблон содержит {{ content }} и заменим его на widget
        template = self.main_template.replace("{{ content }}", markdown_widget)
        
        template_path = os.path.join(self.output_dir, "template.vue")
        with open(template_path, "w") as f:
            f.write(template)
            
        # Выведем информацию о созданном файле для отладки
        logger.info(f"Template.vue generated: {os.path.getsize(template_path)} bytes")
        
        return template_path
    
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

    def _get_report_link(self, template_id: int):
        """Get path to report file."""
        return self.api.server_address + "/nn/experiments/" + str(template_id)

    def _generate_md(self) -> str:
        """Generate content in markdown format.

        :return: markdown content
        :rtype: str
        """
        sections = []

        # Header
        sections.append(self._generate_header())

        # Buttons
        sections.append(self._generate_buttons())
        sections.append("---\n")

        # Overview
        sections.append(self._generate_overview_section())
        sections.append("---\n")

        # Basic information
        sections.append(self._generate_basic_info())

        # Checkpoints
        sections.append(self._generate_checkpoints_section())

        # Predictions
        sections.append(self._generate_predictions_section())

        # Metrics
        sections.append(self._generate_metrics_section())

        # Hyperparameters
        sections.append(self._generate_hyperparameters_section())

        # Training chart
        sections.append(self._generate_training_section())

        # Model API section
        sections.append(self._generate_model_api_section())

        # Docker section
        sections.append(self._generate_docker_section())

        # Local prediction section
        sections.append(self._generate_local_prediction_section())

        return "\n".join(sections)

    def _generate_header(self) -> str:
        """Generate page header.

        :return: Page header content
        :rtype: str
        """
        exp_name = self.info.get("experiment_name", "Experiment")
        return f'# Experiment "{exp_name}"\n\n'

    def _generate_buttons(self) -> str:
        """Generate buttons section.

        :return: Buttons section content
        :rtype: str
        """
        buttons = [
            "## Buttons",
            "- 🚀 Deploy (PyTorch)",
        ]

        # Add TensorRT button if TensorRT export exists
        export = self.info.get("export", {})
        if export and any("engine" in k.lower() for k in export.keys()):
            buttons.append("- 🚀 Deploy (TensorRT)")
        elif export and any("onnx" in k.lower() for k in export.keys()):
            buttons.append("- 🚀 Deploy (ONNX)")

        buttons.extend(
            [
                "- ⏩ Fine-tune",
                "- 🔄 Re-train",
                "- 📦 Download model",
                "- ❌ Remove permamently",
            ]
        )

        return "\n".join(buttons) + "\n"

    def _generate_overview_section(self) -> str:
        """Generate overview section with links.

        :return: Overview section content
        :rtype: str
        """
        lines = ["## Overview\n"]

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
            report_link = (
                eval_link or f"{self.api.server_address}/model-benchmark?id={eval_id}"
            )
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

        return "\n".join(lines) + "\n"

    def _generate_basic_info(self) -> str:
        """Generate section with basic information.

        :return: Basic information section content
        :rtype: str
        """
        lines = []

        # Model name
        model_name = self.info.get("model_name")
        if model_name:
            lines.append(f"- **Model**: {model_name}")

        # Task type
        task_type = self.info.get("task_type")
        if task_type:
            lines.append(f"- **Task**: {task_type}")

        # Framework
        framework = self.info.get("framework_name")
        if framework:
            lines.append(f"- **Framework**: {framework}")

        # Project
        project_id = self.info.get("project_id")
        project_info = self.api.project.get_info_by_id(project_id)
        if project_info:
            lines.append(
                f"- **Project**: [{project_info.name}]({self.api.server_address}/projects/{project_id}/datasets)"
            )

        # Train dataset size
        train_size = self.info.get("train_size")
        if train_size:
            if project_info:
                lines.append(f"- **Train size**: {train_size} {project_info.type}")
            else:
                lines.append(f"- **Train size**: {train_size}")

        # Validation dataset size
        val_size = self.info.get("val_size")
        if val_size:
            if project_info:
                lines.append(f"- **Validation size**: {val_size} {project_info.type}")
            else:
                lines.append(f"- **Validation size**: {val_size}")

        # Classes
        classes = [cls.name for cls in self.model_meta.obj_classes]
        if classes:
            lines.append(f"- **Classes**: {len(classes)}")
            lines.append(f"- **Class names**: {', '.join(classes)}")

        # Tags
        # tags = [tag.name for tag in self.model_meta.tag_metas]
        # if tags:
        #     lines.append(f"- **Tags**: {len(tags)}")
        #     lines.append(f"- **Tag names**: {', '.join(tags)}")

        # Training date
        date_str = self.info.get("datetime")
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                formatted_date = dt.strftime("%d %b %Y")
                lines.append(f"- **Date**: {formatted_date}")
            except ValueError:
                lines.append(f"- **Date**: {date_str}")

        return "\n".join(lines) + "\n\n"

    def _generate_checkpoints_section(self) -> str:
        """Generate checkpoints section.

        :return: Checkpoints section content
        :rtype: str
        """
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

        return "\n".join(lines) + "\n\n"

    def _generate_predictions_section(self) -> str:
        """Generate predictions section.

        :return: Predictions section content
        :rtype: str
        """
        # Check if there is an image with predictions in artifacts
        predictions_path = None
        if self.artifacts_dir:
            for filename in ["predictions.png", "prediction.png", "examples.png", "example.png"]:
                path = os.path.join(self.artifacts_dir, filename)
                if os.path.exists(path):
                    predictions_path = path
                    break

        if not predictions_path:
            return ""

        # Copy to images directory
        dest_path = os.path.join(self.img_dir, "predictions.png")
        if not os.path.exists(dest_path):
            shutil.copy(predictions_path, dest_path)

        return "## Predictions\n\n![predictions](img/predictions.png)\n\n"

    def _generate_metrics_section(self) -> str:
        """Generate metrics section.

        :return: Metrics section content
        :rtype: str
        """
        metrics = self.info.get("evaluation_metrics", {})
        if not metrics:
            return ""

        lines = [
            "## Metrics\n",
            "| Metrics | Values |",
            "|---------|--------|",
        ]

        # Get primary metric if specified
        primary_metric = self.info.get("primary_metric")

        # List of important metrics by task types
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

        return "\n".join(lines) + "\n\n"

    def _generate_hyperparameters_section(self) -> str:
        """Generate hyperparameters section.

        :return: Hyperparameters section content
        :rtype: str
        """
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

        if hyperparams_yaml:
            return f"## Training Hyperparameters\n\n```yaml\n{hyperparams_yaml}```\n\n"
        else:
            return ""

    def _generate_training_section(self) -> str:
        """Generate training chart section.

        :return: Training chart section content
        :rtype: str
        """
        # Check if there's a training chart image
        chart_path = None
        if self.artifacts_dir:
            for filename in ["chart.png", "training.png", "loss.png", "metrics.png"]:
                path = os.path.join(self.artifacts_dir, filename)
                if os.path.exists(path):
                    chart_path = path
                    break

        if not chart_path:
            return ""

        # Copy to images directory
        dest_path = os.path.join(self.img_dir, "chart.png")
        if not os.path.exists(dest_path):
            shutil.copy(chart_path, dest_path)

        return "## Training\n\n![chart](img/chart.png)\n\n"

    def _generate_model_api_section(self) -> str:
        """Generate Model API description section.

        :return: Model API section content
        :rtype: str
        """
        # Check if model files exist
        if not self.info.get("model_files"):
            return ""

        checkpoint_id = "{checkpoint_id}"  # Default placeholder

        return f"""## Model API

Deploy and predict in Supervisely.

```python
import supervisely as sly

api = sly.Api()

# Deploy
model = api.nn.deploy_custom_model(
    checkpoint_id={checkpoint_id},  # file id
)

# Predict
prediction = model.predict(
    images="image.png"  # image | path | url
)
```

> See more in [Deploy and Predict with Supervisely SDK](https://docs.supervisely.com/neural-networks/overview-1/deploy_and_predict_with_supervisely_sdk) documentation.

"""

    def _generate_docker_section(self) -> str:
        """Generate Docker description section.

        :return: Docker section content
        :rtype: str
        """
        framework_name = self.info.get("framework_name")
        if not framework_name:
            return ""

        # Create expected Docker image name
        if self.app_info:
            docker_image = self.app_info.config.get("docker_image")
        else:
            docker_image = f"supervisely/{framework_name.lower()}:latest"
        experiment_dir = os.path.basename(self.info.get("artifacts_dir", "")) or "{experiment_dir}"

        return f"""## Docker

Predict using Docker container.

1. Download checkpoint from Supervisely ([download](xxx))

2. Pull the Docker image

```bash
docker pull {docker_image}
```
3. Run the Docker container

```bash
docker run \\
  --runtime=nvidia \\
  -v "./{experiment_dir}:/model" \\
  -p 8000:8000 \\
  {docker_image} \\
  predict \\
  "./image.jpg" \\
  --model "/model" \\
  --device "cuda:0" \\
```

> See more in [Deploy in Docker Container](https://docs.supervisely.com/neural-networks/overview-1/deploy_and_predict_with_supervisely_sdk#deploy-in-docker-container) documentation.

"""

    def _generate_local_prediction_section(self) -> str:
        """Generate local prediction description section.

        :return: Local prediction section content
        :rtype: str
        """
        framework_name = self.info.get("framework_name")
        if not framework_name:
            return ""

        if self.app_info:
            repo_link = self.app_info.repo
            repo_name = repo_link.split("/")[-1]
        else:
            repo_link = f"https://github.com/supervisely-ecosystem/{framework_name.replace(' ', '-')}"
            repo_name = framework_name.replace(" ", "-")


        model_class = self.serving_class.__name__
        model_class_import_path = f"from {self.serving_class.__module__} import {model_class}"
        experiment_dir = os.path.basename(self.info.get("artifacts_dir", "")) or "{experiment_dir}"

        # Check if there's export to ONNX or TensorRT
        export = self.info.get("export", {})
        has_onnx = export and any("onnx" in k.lower() for k in export.keys())
        has_engine = export and any("engine" in k.lower() for k in export.keys())

        onnx_section = ""
        if has_onnx or has_engine:
            checkpoint_name = "best.onnx" if has_onnx else "best.engine"
            onnx_section = f"""

Questions:
- How to load {checkpoint_name}?

```python
# Be sure you are in the root of the {repo_name} repository
{model_class_import_path}

model = {model_class}(
    model_dir="./{experiment_dir}",
    checkpoint="{checkpoint_name}",
    device="cuda",
)
```"""

        return f"""## Predict Locally

1. Download checkpoint from Supervisely ([download](xxx))

2. Clone repository

```bash
git clone {repo_link}
cd {repo_name}
```

3. Install requirements

```bash
pip install -r dev_requirements.txt
pip install supervisely
```

4. Run inference

```python
# Be sure you are in the root of the {repo_name} repository
{model_class_import_path}

# Load model
model = {model_class}(
    checkpoint="./{experiment_dir}/checkpoints/{self.info.get('best_checkpoint', 'best.pt')}",  # path to the checkpoint
    device="cuda",
)

# Predict
prediction = model(
    "image.png",  # local paths, directory, local project, np.array, PIL.Image, URL
    params={{"confidence_threshold": 0.5}}
)
```{onnx_section}
"""

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