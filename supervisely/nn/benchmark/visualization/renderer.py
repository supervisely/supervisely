import json
from pathlib import Path
from typing import Optional

from jinja2 import Template

from supervisely.api.api import Api
from supervisely.io.fs import dir_empty, get_directory_size
from supervisely.nn.benchmark.visualization.widgets import BaseWidget
from supervisely.task.progress import tqdm_sly
from supervisely import logger


class Renderer:
    """
    Base class for rendering visualizations of Evaluation Report.
    """

    def __init__(
        self,
        layout: BaseWidget,
        base_dir: str = "./output",
        template: str = None,
        report_name: str = "Model Evaluation Report.lnk",
    ) -> None:
        if template is None:
            template = (
                Path(__file__).parents[1].joinpath("visualization/report_template.html").read_text()
            )
        self.main_template = template
        self.layout = layout
        self.base_dir = base_dir
        self.report_name = report_name
        self._report = None
        self._lnk = None

        if Path(base_dir).exists():
            if not dir_empty(base_dir):
                raise ValueError(f"Output directory {base_dir} is not empty.")

    @property
    def _template_data(self):
        return {"layout": self.layout.to_html()}

    def render(self):
        return Template(self.main_template).render(self._template_data)

    def get_state(self):
        return {}

    def save(self) -> None:
        self.layout.save_data(self.base_dir)
        state = self.layout.get_state()
        with open(Path(self.base_dir).joinpath("state.json"), "w") as f:
            json.dump(state, f)
        template = self.render()
        with open(Path(self.base_dir).joinpath("template.vue"), "w") as f:
            f.write(template)
        return template

    def visualize(self):
        return self.save()

    def upload_results(
        self, api: Api, team_id: int, remote_dir: str, progress: Optional[tqdm_sly] = None
    ) -> str:
        if dir_empty(self.base_dir):
            raise RuntimeError(
                "No visualizations to upload. You should call visualize method first."
            )
        if progress is None:
            progress = tqdm_sly
        dir_total = get_directory_size(self.base_dir)
        dir_name = Path(remote_dir).name
        with progress(
            message=f"Uploading visualizations to {dir_name}",
            total=dir_total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            remote_dir = api.file.upload_directory(
                team_id,
                self.base_dir,
                remote_dir,
                change_name_if_conflict=True,
                progress_size_cb=pbar,
            )
        src = self._save_report_link(api, team_id, remote_dir)
        dst = Path(remote_dir).joinpath(self.report_name)
        self._lnk = api.file.upload(team_id=team_id, src=src, dst=str(dst))
        return remote_dir

    def _save_report_link(self, api: Api, team_id: int, remote_dir: str):
        report_link = self._get_report_path(api, team_id, remote_dir)
        pth = Path(self.base_dir).joinpath(self.report_name)
        with open(pth, "w") as f:
            f.write(report_link)
        logger.debug(f"Report link: {self._get_report_link(api, team_id, remote_dir)}")
        return str(pth)

    def _get_report_link(self, api: Api, team_id: int, remote_dir: str):
        path = self._get_report_path(api, team_id, remote_dir)
        return f"{api.server_address}{path}"

    def _get_report_path(self, api: Api, team_id: int, remote_dir: str):
        template_path = Path(remote_dir).joinpath("template.vue")
        self._report = api.file.get_info_by_path(team_id, str(template_path))
        return "/model-benchmark?id=" + str(self._report.id)

    @property
    def report(self):
        return self._report

    @property
    def lnk(self):
        return self._lnk
