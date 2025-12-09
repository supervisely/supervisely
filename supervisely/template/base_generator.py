import inspect
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
    Base class for generating reports from Jinja2 templates and uploading them to Supervisely.
    """

    TEMPLATE = "template.html.jinja"
    VUE_TEMPLATE_NAME = "template.vue"
    LINK_FILE = "Open Report.lnk"

    def __init__(self, api: Api, output_dir: str):
        self.api = api
        self.output_dir = output_dir
        self.template_renderer = TemplateRenderer()
        os.makedirs(self.output_dir, exist_ok=True)
    
    @property
    def template_path(self) -> str:
        cls_dir = Path(inspect.getfile(self.__class__)).parent
        return f"{cls_dir}/{self.TEMPLATE}"

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
        self.api.file.upload_directory_fast(
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
        if self._report_url(self.api.server_address, template_id) is not None:
            url = self._upload_link_file(template_id, remote_dir, team_id)
            logger.info(f"Open URL: {url}")
        else:
            logger.warning("Subclasses must implement the `_report_url` method to upload a link file.")
        return template_id
    
    def _render(self) -> str:
        context = self.context()
        content = self.template_renderer.render(self.template_path, context)
        return content

    def _upload_link_file(self, template_id: int, remote_dir: str, team_id: int):
        url = self._report_url(self.api.server_address, template_id)
        link_path = os.path.join(self.output_dir, self.LINK_FILE)
        with open(link_path, "w") as f:
            f.write(url)
        self.api.file.upload(team_id=team_id, src=link_path, dst=self._link_file_dst_path(remote_dir))
        return url
        
    def _link_file_dst_path(self, remote_dir: str) -> str:
        return f"{remote_dir}/{self.LINK_FILE}"

    def _report_url(self, server_address: str, template_id: int) -> str:
        raise NotImplementedError("Subclasses must implement the `_report_url` method to upload a link file.")