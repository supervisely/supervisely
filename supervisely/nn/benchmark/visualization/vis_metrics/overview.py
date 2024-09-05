from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class Overview(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self._is_overview = True
        info = loader.inference_info
        url = info.get("checkpoint_url")
        link_text = info.get("custom_checkpoint_path")
        if link_text is None:
            link_text = url
        link_text = link_text.replace("_", "\_")
        
        # Note about validation dataset
        gt_project_id = self._loader.gt_project_info.id
        gt_images_ids = self._loader._benchmark.gt_images_ids
        gt_dataset_ids = self._loader._benchmark.gt_dataset_ids
        if gt_images_ids is not None:
            note_about_val_dataset = "Evaluated using validation subset."
        elif gt_dataset_ids is not None:
            links = []
            for gt_dataset_id in gt_dataset_ids:
                gt_dataset_name = self._loader._api.dataset.get_info_by_id(gt_dataset_id).name
                link = f'<a href="/projects/{gt_project_id}/datasets/{gt_dataset_id}" target="_blank">{gt_dataset_name}</a>'
                links.append(link)
            if len(links) == 1:
                note_about_val_dataset = f"Evaluated on the validation dataset: {links[0]}"
            else:
                note_about_val_dataset = f"Evaluated on the validation datasets: {', '.join(links)}"
        else:
            note_about_val_dataset = ""
        if note_about_val_dataset:
            note_about_val_dataset = "\n" + note_about_val_dataset + "\n"
        
        checkpoint_name = info.get("deploy_params", {}).get("checkpoint_name", "").replace("_", "\_")
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_overview=Widget.Markdown(
                title="Overview",
                is_header=True,
                formats=[
                    checkpoint_name,  # Title
                    info.get("model_name"),
                    checkpoint_name,
                    info.get("architecture"),
                    info.get("task_type"),
                    info.get("runtime"),
                    url,
                    link_text,
                    self._loader.gt_project_info.id,
                    self._loader.gt_project_info.name,
                    note_about_val_dataset,
                    self._loader.docs_link,
                ],
            ),
            markdown_key_metrics=Widget.Markdown(
                title="Key Metrics",
                is_header=True,
                formats=[
                    self._loader.vis_texts.definitions.average_precision,
                    self._loader.vis_texts.definitions.confidence_threshold,
                    self._loader.vis_texts.definitions.confidence_score,
                ],
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error

        # Overall Metrics
        base_metrics = self._loader.mp.base_metrics()
        r = list(base_metrics.values())
        theta = [self._loader.mp.metric_names[k] for k in base_metrics.keys()]
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=r + [r[0]],
                theta=theta + [theta[0]],
                fill="toself",
                name="Overall Metrics",
                hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0.0, 1.0],
                    ticks="outside",
                ),
                angularaxis=dict(rotation=90, direction="clockwise"),
            ),
            dragmode=False,
            # title="Overall Metrics",
            # width=700,
            # height=500,
            # autosize=False,
            margin=dict(l=25, r=25, t=25, b=25),
        )
        fig.update_layout(
            modebar=dict(
                remove=[
                    "zoom2d",
                    "pan2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ]
            )
        )
        return fig

    def get_main_info(self) -> tuple:
        inference_info = self._loader.inference_info
        title = inference_info.get("deploy_params", {}).get("checkpoint_name", "")
        title = title.replace("_", "\_")
        me = self._loader._api.user.get_my_info().login
        current_date = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        return title, me, current_date
