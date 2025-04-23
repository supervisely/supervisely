import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
from supervisely import logger
from supervisely.api.api import Api
from supervisely.template.template_renderer import TemplateRenderer


class BaseGenerator:
    """
    Base class for generating reports from Jinja2 templates.
    """

    URL_NAME = None
    LINK_FILE = "Open Report.lnk"
    VUE_TEMPLATE_NAME = "template.vue"

    def __init__(self, api: Api, template_path: str, output_dir: str):
        self.api = api
        self.template_path = template_path
        self.output_dir = output_dir
        self.template_renderer = TemplateRenderer(self.template_path)
        os.makedirs(self.output_dir, exist_ok=True)

    def context(self) -> dict:
        raise NotImplementedError("Subclasses must implement the context method.")
    
    def state(self) -> dict:
        return {}
        
    def generate(self):
        # Render
        content = self._render()
        # Save template.vue
        template_path = f"{self.output_dir}/{self.VUE_TEMPLATE_NAME}"
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(content)
        # Save state.json
        state = self.state()
        state_path = f"{self.output_dir}/state.json"
        sly_json.dump_json_file(state, state_path)
    
    def upload(self, remote_dir: str, team_id: Optional[int] = None, **kwargs):
        team_id = team_id or sly_env.team_id()
        remote_dir = self.api.file.upload_directory_fast(
            team_id=team_id,
            local_dir=self.output_dir,
            remote_dir=remote_dir,
            **kwargs
        )
        logger.info(f"Template uploaded to {remote_dir}")
        template_id = self.api.file.get_info_by_path(
            team_id=team_id,
            remote_path=f"{remote_dir}/{self.VUE_TEMPLATE_NAME}",
        ).id
        link_path = self._generate_link_file(template_id)
        self._upload_link_file(remote_dir, link_path, team_id)
        logger.info(f"Open URL: {self._get_report_url(template_id)}")
        return template_id
    
    def _render(self) -> str:
        context = self.context()
        content = self.template_renderer.render(self.template_path, context)
        return content

    def _upload_link_file(self, remote_dir: str, link_path: str, team_id: int):
        self.api.file.upload(team_id=team_id, src=link_path, dst=f"{remote_dir}/{self.LINK_FILE}")
        
    def _generate_link_file(self, template_id: int) -> str:
        url = self._get_report_url(template_id)
        open_link_path = os.path.join(self.output_dir, self.LINK_FILE)
        with open(open_link_path, "w") as f:
            f.write(url)
        return open_link_path

    def _get_report_url(self, template_id: int) -> str:
        return f"{self.api.server_address}/{self.URL_NAME}&id={template_id}"