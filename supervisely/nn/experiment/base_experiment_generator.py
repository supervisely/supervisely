import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional, Union

import yaml

from supervisely.io.fs import mkdir
from supervisely.nn.inference import Inference
from supervisely.project import ProjectMeta


class BaseExperimentGenerator:
    """
    Base class for generating experiment reports.

    Provides functionality for generating experiment report pages in Markdown format.
    """

    def __init__(
        self,
        experiment_info: dict,
        hyperparameters: str,
        model_meta: ProjectMeta,
        serving_class: Optional[Inference] = None,
        output_dir: str = "./experiment_report",
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
        :param output_dir: Directory to save the report
        :type output_dir: str
        """
        self.info = experiment_info
        self.hyperparameters = hyperparameters
        self.model_meta = model_meta
        self.output_dir = output_dir
        self.artifacts_dir = self.info["artifacts_dir"]
        self.serving_class = serving_class

        # @TODO: remove this
        self.img_dir = os.path.join(output_dir, "img")
        mkdir(self.img_dir, True)

    def generate_report(self) -> str:
        """Generate experiment report and return path to README.md.

        :return: Path to the generated README.md file
        :rtype: str
        """
        mkdir(self.output_dir, True)

        # Generate README.md content
        content = self._generate_content()
        readme_path = os.path.join(self.output_dir, "README.md")

        # Write to file
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

        return readme_path

    def _generate_content(self) -> str:
        """Generate content for README.md file.

        :return: README.md content
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
                f"- 🎓 [Training Task](https://dev.internal.supervisely.com/apps/sessions/{task_id})"
            )

        # Evaluation report link
        eval_id = self.info.get("evaluation_report_id")
        eval_link = self.info.get("evaluation_report_link")
        if eval_id:
            report_link = (
                eval_link or f"https://dev.internal.supervisely.com/model-benchmark?id={eval_id}"
            )
            lines.append(f"- 📊 [Evaluation Report]({report_link})")

        # TensorBoard logs link
        logs = self.info.get("logs", {})
        if logs and "link" in logs:
            lines.append(f"- ⚡ [TensorBoard Logs]({logs['link']})")

        # Artifacts link in Team Files
        artifacts_dir = self.info.get("artifacts_dir")
        if artifacts_dir:
            lines.append(
                f"- 💾 [Open in Team Files](https://dev.internal.supervisely.com/files/?path={artifacts_dir})"
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
        if project_id:
            lines.append(
                f"- **Project**: [Project {project_id}](https://dev.internal.supervisely.com/projects/{project_id}/datasets)"
            )

        # Train dataset size
        train_size = self.info.get("train_size")
        if train_size:
            lines.append(f"- **Train dataset size**: {train_size} images")

        # Validation dataset size
        val_size = self.info.get("val_size")
        if val_size:
            lines.append(f"- **Validation dataset size**: {val_size} images")

        # Classes
        classes = [cls.name for cls in self.model_meta.obj_classes]
        if classes:
            lines.append(f"- **Classes**: {len(classes)}")
            lines.append(f"- **Class names**: {', '.join(classes)}")

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

        # Create expected repository and model class names
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
from supervisely_integration.serve.{model_class.lower()} import {model_class}
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
git clone https://github.com/supervisely-ecosystem/{repo_name}
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
from supervisely_integration.serve.{model_class.lower()} import {model_class}

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
