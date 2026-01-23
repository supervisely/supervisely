from __future__ import annotations

import os
import re
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.subplots import make_subplots  # pylint: disable=import-error

import supervisely as sly
from supervisely import Api, ProjectMeta, logger
from supervisely.template.base_generator import BaseGenerator
from supervisely.imaging.color import rgb2hex


class LiveTrainingGenerator(BaseGenerator):
    """
    Generator for Live training session reports.
    
    Logs:
    - Model hyperparameters
    - Training loss graphs
    - Checkpoints
    - Dataset size over time
    """

    def __init__(
        self,
        api: Api,
        session_info: dict,
        model_config: dict,
        model_meta: ProjectMeta,
        task_type: str,
        output_dir: str = "./live_training_report",
        team_id: Optional[int] = None,

    ):
        """
        Initialize Live training generator.
        
        :param api: Supervisely API instance
        :param session_info: Session metadata (session_id, start_time, project_id, etc.)
        :param model_config: Model configuration (hyperparameters, backbone, etc.)
        :param model_meta: Model metadata with classes
        :param output_dir: Local output directory
        :param team_id: Team ID
        """
        super().__init__(api, output_dir)
        self.team_id = team_id or sly.env.team_id()
        self.session_info = session_info
        self.model_config = model_config
        self.model_meta = model_meta
        self.task_type = task_type
        self._slug_map = {
            "semantic segmentation": "supervisely-ecosystem/live-training---semantic-segmentation",
            "object detection": "supervisely-ecosystem/live-training---object-detection",
        }
        self.slug = self._slug_map[task_type] 
        
        # Validate required fields
        self._validate_session_info()

    def _validate_session_info(self):
        """Validate that session_info contains required fields"""
        required = ["session_id", "project_id", "start_time"]
        missing = [k for k in required if k not in self.session_info]
        if missing:
            raise ValueError(f"Missing required fields in session_info: {missing}")

    def _report_url(self, server_address: str, template_id: int) -> str:
        """Generate URL to open the Live training report"""
        return f"{server_address}/nn/experiments/{template_id}"

    def context(self) -> dict:
        return {
            "env": self._get_env_context(),
            "session": self._get_session_context(),
            "model": self._get_model_context(),
            "training": self._get_training_context(),
            "dataset": self._get_dataset_context(),
            "widgets": self._get_widgets_context(), 
            "resources": self._get_resources_context(),
        }
    
    def _get_env_context(self) -> dict:
        """Environment info"""
        return {
            "server_address": self.api.server_address,
        }

    def _get_session_context(self) -> dict:
        session_id = self.session_info["session_id"]
        project_id = self.session_info["project_id"]
        artifacts_dir = self.session_info.get("artifacts_dir", "")
        task_id = self.session_info.get("task_id", session_id)
        
        project_info = self.api.project.get_info_by_id(project_id)
        project_url = f"{self.api.server_address}/projects/{project_id}/datasets"
        artifacts_url = f"{self.api.server_address}/files/?path={artifacts_dir}" if artifacts_dir else None
        
        return {
            "id": session_id,
            "task_id": task_id,
            "name": self.session_info.get("session_name", f"Session {session_id}"),
            "start_time": self.session_info["start_time"],
            "duration": self.session_info.get("duration"),
            "current_iteration": self.session_info.get("current_iteration", 0),
            "artifacts_url": artifacts_url,
            "artifacts_dir": artifacts_dir,  
            "project": {
                "id": project_id,
                "name": project_info.name if project_info else "Unknown",
                "url": project_url if project_info else None,
            },
            "status": self.session_info.get("status", "running"),
        }
    
    @staticmethod
    def parse_hyperparameters(config_path: str) -> dict:
        """
        Parse hyperparameters from MMEngine config file.
        
        :param config_path: Path to config.py
        :return: Dict with extracted hyperparameters
        """
        # TODO: only basic parsing for segmentation
        hyperparams = {}
        
        if not os.path.exists(config_path):
            return hyperparams
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Extract crop_size
        match = re.search(r'crop_size\s*=\s*\((\d+),\s*(\d+)\)', content)
        if match:
            hyperparams['crop_size'] = f"({match.group(1)}, {match.group(2)})"
        
        # Extract learning rate
        match = re.search(r'lr=([0-9.e-]+)', content)
        if match:
            hyperparams['learning_rate'] = float(match.group(1))
        
        # Extract batch_size
        match = re.search(r'batch_size=(\d+)', content)
        if match:
            hyperparams['batch_size'] = int(match.group(1))
        
        # Extract max_epochs
        match = re.search(r'max_epochs\s*=\s*(\d+)', content)
        if match:
            hyperparams['max_epochs'] = int(match.group(1))
        
        # Extract weight_decay
        match = re.search(r'weight_decay=([0-9.e-]+)', content)
        if match:
            hyperparams['weight_decay'] = float(match.group(1))
        
        # Extract optimizer
        match = re.search(r"optimizer=dict\(type='(\w+)'", content)
        if match:
            hyperparams['optimizer'] = match.group(1)
        
        return hyperparams

    def _get_model_context(self) -> dict:
        """Model configuration info"""
        classes = [cls.name for cls in self.model_meta.obj_classes if cls.name != "_background_"]    
        display_name = self.model_config.get("display_name", self.model_config.get("model_name", "Unknown"))

        return {
            "name": display_name,
            "backbone": self.model_config.get("backbone", "N/A"),
            "num_classes": len(classes),
            "classes": classes,
            "classes_short": classes[:3] + (["..."] if len(classes) > 3 else []),
            "config_file": self.model_config.get("config_file", "N/A"),
            "task_type": self.model_config.get("task_type", "Live Training"),
        }

    def _get_training_context(self) -> dict:
        """Training logs and checkpoints"""
        logs_path = self.session_info.get("logs_dir")
        logs_url = None
        if logs_path:
            logs_url = f"{self.api.server_address}/files/?path={logs_path}"
        
        checkpoints = []
        artifacts_dir = self.session_info.get("artifacts_dir", "")
        for ckpt in self.session_info.get("checkpoints", []):
            checkpoint = {
                "name": ckpt["name"],
                "iteration": ckpt["iteration"],
                "loss": ckpt.get("loss"),
                "url": f"{self.api.server_address}/files/?path={artifacts_dir}/checkpoints/{ckpt['name']}",
            }
            checkpoints.append(checkpoint)
        
        # Get total iterations from loss_history or checkpoints
        loss_history = self.session_info.get("loss_history", [])
        # Handle both old (list) and new (dict) formats
        if isinstance(loss_history, list) and loss_history:
            total_iterations = loss_history[-1]["iteration"]
        elif isinstance(loss_history, dict):
            # Get max step from any metric
            total_iterations = max(
                (item["step"] for metric_data in loss_history.values() for item in metric_data),
                default=0
            ) if loss_history else 0
        else:
            total_iterations = max([c["iteration"] for c in self.session_info.get("checkpoints", [])]) if self.session_info.get("checkpoints") else 0
            
        
        return {
            "total_iterations": total_iterations,
            "device": self.session_info.get("device", "N/A"),
            "session_url": self.session_info.get("session_url"),
            "checkpoints": checkpoints,
            "logs": {
                "path": logs_path,
                "url": logs_url,
            },
        }
     
    def _get_dataset_context(self) -> dict:
        """Dataset info"""
        return {
            "current_size": self.session_info.get("dataset_size", 0),
            "initial_samples": self.session_info.get("initial_samples", 0),
        }

    def _get_training_plots_html(self) -> Optional[str]:
        """
        Generate HTML for training loss plot.
        Currently returns None - to be implemented later with actual loss data.
        """
        # TODO: Generate plot from loss history
        # For now return placeholder
        return None

    def _generate_classes_table(self) -> str:
        """Generate HTML table with class names, shapes and colors.

        :returns: HTML string with classes table
        :rtype: str
        """
        type_to_icon = {
            sly.AnyGeometry: "zmdi zmdi-shape",
            sly.Rectangle: "zmdi zmdi-crop-din",
            sly.Polygon: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAB6klEQVRYhe2Wuy8EURTGf+u5VESNXq2yhYZCoeBv8RcI1i6NVUpsoVCKkHjUGlFTiYb1mFmh2MiKjVXMudmb3cPOzB0VXzKZm5k53/nmvO6Ff4RHD5AD7gFP1l3Kd11AHvCBEpAVW2esAvWmK6t8l1O+W0lCQEnIJoAZxUnzNQNkZF36jrQjgoA+uaciCgc9VaExBOyh/6WWAi1VhbjOJ4FbIXkBtgkK0BNHnYqNKUIPeBPbKyDdzpld5T6wD9SE4AwYjfEDaXFeFzE/doUWuhqwiFsOCwqv2hV2lU/L+sHBscGTxdvSFVoXpAjCZdauMHVic6ndl6U1VBsJCFhTeNUU9IiIEo3qvQYGHAV0AyfC5wNLhKipXuBCjA5wT8WxcM1FMRoBymK44CjAE57hqIazwCfwQdARcXa3UXHuRXVucIjb7jYvNkdxBZg0TBFid7PQTRAtX2xOiXkuMAMqYwkIE848rZFbjyNAmw9bIeweaZ2A5TgC7PnwKkTPtN+cTOrsyN3FEWAjRTAX6sA5ek77gSL6+WHZVQDAIHAjhJtN78aAS3lXAXYIivBOnCdyOAUYB6o0xqsvziry7FLE/Cp20cNcJEjDr8MUmVOVRzkVN+Nd7vZGVXXgiwxtPiRS5WFhz4fEq/zv4AvToMn7vCn3eAAAAABJRU5ErkJggg==",
            sly.Bitmap: "zmdi zmdi-brush",
            sly.Polyline: "zmdi zmdi-gesture",
            sly.Point: "zmdi zmdi-dot-circle-alt",
            sly.Cuboid: "zmdi zmdi-ungroup",  #
            sly.GraphNodes: "zmdi zmdi-grain",
            sly.MultichannelBitmap: "zmdi zmdi-layers",
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

    def upload_to_artifacts(self, remote_dir: str):
        """
        Upload report to team files.

        Default path: /live-training/{project_id}_{project_name}/{session_id}/
        """
        # Normalize path - remove trailing slash
        remote_dir = remote_dir.rstrip("/")
        file_info = self.upload(remote_dir, team_id=self.team_id)
        self._report_file_info = file_info 
        return file_info 

    def _get_widgets_context(self) -> dict:
        """Generate widgets (tables, plots) for the report"""
        checkpoints_table = self._generate_checkpoints_table()
        training_plot = self._generate_training_plot()
        classes = self._generate_classes_table()
        
        return {
            "tables": {
                "checkpoints": checkpoints_table,
                "classes": classes,
            },
            "training_plot": training_plot,
        }

    def _generate_checkpoints_table(self) -> Optional[str]:
        """Generate HTML table with checkpoints"""
        # Get training context to access checkpoints with URLs
        training_ctx = self._get_training_context()
        checkpoints = training_ctx.get("checkpoints", [])
        
        if not checkpoints:
            return None
        
        html = ['<table class="table">']
        html.append("<thead><tr><th>Checkpoint Name</th><th>Iteration</th><th>Loss</th><th>Actions</th></tr></thead>")
        html.append("<tbody>")
        
        for checkpoint in checkpoints:
            name = checkpoint.get("name", "N/A")
            iteration = checkpoint.get("iteration", "N/A")
            loss = checkpoint.get("loss")
            url = checkpoint.get("url", "")
            loss_str = f"{loss:.6f}" if loss is not None else "N/A"
            
            download_link = f'<a href="{url}" target="_blank" class="download-link">Download</a>' if url else ""
            
            html.append(f"<tr><td>{name}</td><td>{iteration}</td><td>{loss_str}</td><td>{download_link}</td></tr>")
        
        html.append("</tbody>")
        html.append("</table>")
        return "\n".join(html)

    def _generate_training_plot(self) -> str:
        """Generate training plots grid (like Experiments)"""
        loss_history = self.session_info.get("loss_history", {})
        
        if not loss_history or not isinstance(loss_history, dict):
            return "<p>No training data available yet.</p>"
        
        # Get all metrics
        metrics = list(loss_history.keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            return "<p>No training data available yet.</p>"
        
        # Calculate grid size (like in Experiments)
        side = min(4, max(2, math.ceil(math.sqrt(n_metrics))))
        cols = side
        rows = math.ceil(n_metrics / cols)
        
        # Create subplots
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=metrics)
        
        for idx, metric in enumerate(metrics, start=1):
            data = loss_history[metric]
            if not data:
                continue
                
            steps = [item["step"] for item in data]
            values = [item["value"] for item in data]
            
            row = (idx - 1) // cols + 1
            col = (idx - 1) % cols + 1
            
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=values,
                    mode="lines",
                    name=metric,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            
            # Special formatting for training rate
            if metric.startswith("lr"):
                fig.update_yaxes(tickformat=".0e", row=row, col=col)
        
        fig.update_layout(
            height=300 * rows,
            width=400 * cols,
            showlegend=False,
        )
        
        # Save as PNG
        data_dir = os.path.join(self.output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        img_path = os.path.join(data_dir, "training_plots_grid.png")
        
        try:
            fig.write_image(img_path, engine="kaleido")
        except Exception as e:
            logger.warning(f"Failed to save training plot: {e}")
            return "<p>Failed to generate training plot</p>"
        
        # Return Vue image component
        return f'<sly-iw-image src="/data/training_plots_grid.png" :template-base-path="templateBasePath" :options="{{ style: {{ width: \'70%\', height: \'auto\' }} }}" />'
  
    def _get_online_training_app_info(self):
        """Get online training app info from ecosystem"""
        try:
            # TODO: only works for public apps.
            # Exception handles only private apps on dev server. Need implement for private apps on any server.
            module_id = self.api.app.get_ecosystem_module_id(self.slug)
        except Exception as e:
            logger.warning(f"Failed to get module ID for slug {self.slug}: {e}.")
            if self.api.server_address.endswith("dev.internal.supervisely.com"):
                logger.warning("Using hardcoded module ID for dev server")
                task2module_map = {
                    "object detection": 620,
                    "semantic segmentation": 621,
                }
                module_id = task2module_map.get(self.task_type)
            else:
                raise e
        return {
            "slug": self.slug,
            "module_id": module_id,
        }  

    def _get_resources_context(self):
        """Return apps module IDs for buttons"""
        online_training_app = self._get_online_training_app_info()
        
        return {
            "apps": {
                "online_training": online_training_app,
            }
        }

    def get_report(self) -> str:
        """Get report URL after upload"""
        if self._report_file_info is None:
            raise RuntimeError("Report not uploaded yet. Call upload_to_artifacts() first.")
        
        # self._report_file_info is file_id (int), not FileInfo object
        file_id = self._report_file_info if isinstance(self._report_file_info, int) else self._report_file_info.id
        return self._report_url(self.api.server_address, file_id)

    def get_report_id(self) -> int:
        """Get report file ID"""
        if self._report_file_info is None:
            raise RuntimeError("Report not uploaded yet. Call upload_to_artifacts() first.")
        return self._report_file_info.id