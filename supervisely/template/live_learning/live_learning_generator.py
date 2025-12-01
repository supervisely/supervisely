from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import supervisely as sly
from supervisely import Api, ProjectMeta, logger
from supervisely.template.base_generator import BaseGenerator


class LiveLearningGenerator(BaseGenerator):
    """
    Generator for Live Learning session reports.
    
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
        output_dir: str = "./live_learning_report",
        team_id: Optional[int] = None,
    ):
        """
        Initialize Live Learning generator.
        
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
        
        # Validate required fields
        self._validate_session_info()

    def _validate_session_info(self):
        """Validate that session_info contains required fields"""
        required = ["session_id", "project_id", "start_time"]
        missing = [k for k in required if k not in self.session_info]
        if missing:
            raise ValueError(f"Missing required fields in session_info: {missing}")

    def _report_url(self, server_address: str, template_id: int) -> str:
        """Generate URL to open the Live Learning report"""
        return f"{server_address}/nn/live-learning/{template_id}"

    def context(self) -> dict:
        """
        Generate context for Jinja2 template.
        
        Structure:
        - env: server_address
        - session: session metadata
        - model: model configuration
        - training: training logs and checkpoints
        - dataset: dataset info
        """
        return {
            "env": self._get_env_context(),
            "session": self._get_session_context(),
            "model": self._get_model_context(),
            "training": self._get_training_context(),
            "dataset": self._get_dataset_context(),
        }

    def _get_env_context(self) -> dict:
        """Environment info"""
        return {
            "server_address": self.api.server_address,
        }

    def _get_session_context(self) -> dict:
        """Live Learning session info"""
        session_id = self.session_info["session_id"]
        project_id = self.session_info["project_id"]
        
        project_info = self.api.project.get_info_by_id(project_id)
        project_url = f"{self.api.server_address}/projects/{project_id}/datasets"
        
        return {
            "id": session_id,
            "name": self.session_info.get("session_name", f"Session {session_id}"),
            "start_time": self.session_info["start_time"],
            "current_iteration": self.session_info.get("current_iteration", 0),
            "project": {
                "id": project_id,
                "name": project_info.name if project_info else "Unknown",
                "url": project_url if project_info else None,
            },
            "status": self.session_info.get("status", "running"),
        }

    def _get_model_context(self) -> dict:
        """Model configuration info"""
        return {
            "name": self.model_config.get("model_name", "Unknown"),
            "backbone": self.model_config.get("backbone_type", "N/A"),
            "num_classes": self.model_config.get("num_classes", len(self.model_meta.obj_classes)),
            "classes": [cls.name for cls in self.model_meta.obj_classes],
            "config_file": self.model_config.get("config_file", "N/A"),
        }

    def _get_training_context(self) -> dict:
        """Training logs and checkpoints"""
        logs_path = self.session_info.get("logs_dir")
        logs_url = None
        if logs_path:
            logs_url = f"{self.api.server_address}/files/?path={logs_path}"
        
        checkpoints = self.session_info.get("checkpoints", [])
        
        return {
            "current_iteration": self.session_info.get("current_iteration", 0),
            "current_loss": self.session_info.get("current_loss", None),
            "is_paused": self.session_info.get("training_paused", False),
            "checkpoints": checkpoints,
            "logs": {
                "path": logs_path,
                "url": logs_url,
            },
            "training_plots": self._get_training_plots_html(),
        }

    def _get_dataset_context(self) -> dict:
        """Dataset info"""
        return {
            "current_size": self.session_info.get("dataset_size", 0),
            "initial_samples": self.session_info.get("initial_samples", 0),
            "samples_added": self.session_info.get("samples_added", 0),
        }

    def _get_training_plots_html(self) -> Optional[str]:
        """
        Generate HTML for training loss plot.
        Currently returns None - to be implemented later with actual loss data.
        """
        # TODO: Generate plot from loss history
        # For now return placeholder
        return None

    def upload_to_artifacts(self, remote_dir: str):
        """
        Upload report to team files.
        
        Default path: /live-learning/{project_id}_{project_name}/{session_id}/
        """
        self.upload(remote_dir, team_id=self.team_id)

    def get_report(self):
        """Get FileInfo of the uploaded report"""
        artifacts_dir = self.session_info.get("artifacts_dir")
        if not artifacts_dir:
            raise ValueError("artifacts_dir not set in session_info")
        
        remote_report_path = os.path.join(artifacts_dir, "template.vue")
        report_info = self.api.file.get_info_by_path(self.team_id, remote_report_path)
        if report_info is None:
            raise ValueError("Generate and upload report first")
        return report_info

    def get_report_id(self) -> int:
        """Get report file ID"""
        return self.get_report().id

    def get_report_link(self) -> str:
        """Get report URL"""
        return self._report_url(self.api.server_address, self.get_report_id())