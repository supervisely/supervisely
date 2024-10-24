import json
from pathlib import Path
from typing import Optional

from jinja2 import Template

from supervisely.api.api import Api
from supervisely.io.fs import dir_empty, get_directory_size
from supervisely.nn.benchmark.visualization.widgets import BaseWidget
from supervisely.task.progress import tqdm_sly


class Renderer:
    """
    Base class for rendering visualizations of Evaluation Report.
    """

    def __init__(
        self,
        layout: BaseWidget,
        base_dir: str = "./output",
        template: str = None,
    ) -> None:
        if template is None:
            template = (
                Path(__file__).parents[1].joinpath("visualization/report_template.html").read_text()
            )
        self.main_template = template
        self.layout = layout
        self.base_dir = base_dir

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
        src = self.save_report_link(api, team_id, remote_dir)
        api.file.upload(team_id=team_id, src=src, dst=remote_dir.rstrip("/") + "/open.lnk")
        return remote_dir

    def save_report_link(self, api: Api, team_id: int, remote_dir: str):
        report_link = self.get_report_link(api, team_id, remote_dir)
        pth = Path(self.base_dir).joinpath("open.lnk")
        with open(pth, "w") as f:
            f.write(report_link)
        return str(pth)

    def get_report_link(self, api: Api, team_id: int, remote_dir: str):
        template_path = remote_dir.rstrip("/") + "/" + "template.vue"
        vue_template_info = api.file.get_info_by_path(team_id, template_path)

        report_link = "/model-benchmark?id=" + str(vue_template_info.id)
        return report_link
