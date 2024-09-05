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
        classes_str, note_about_val_dataset, train_session = self.get_overview_info()

        checkpoint_name = (
            info.get("deploy_params", {}).get("checkpoint_name", "").replace("_", "\_")
        )
        me = self._loader._api.user.get_my_info().login
        current_date = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_header=Widget.Markdown(
                title="Header",
                is_header=False,
                formats=[checkpoint_name, me, current_date],  # Title
            ),
            markdown_overview=Widget.Markdown(
                title="Overview",
                is_header=True,
                formats=[
                    info.get("model_name"),
                    checkpoint_name,
                    info.get("architecture"),
                    info.get("task_type"),
                    info.get("runtime"),
                    url,
                    link_text,
                    self._loader.gt_project_info.id,
                    self._loader.gt_project_info.name,
                    classes_str,
                    note_about_val_dataset,
                    train_session,
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

    def get_overview_info(self):
        classes_cnt = len(self._loader._benchmark.classes_whitelist)
        classes_str = "classes" if classes_cnt > 1 else "class"
        classes_str = f"{classes_cnt} {classes_str}"
        train_session, note_about_val_dataset = "", ""

        train_info = self._loader._benchmark.train_info
        if train_info:
            train_task_id = train_info.get("app_session_id")
            if train_task_id:
                task_info = self._loader._api.task.get_info_by_id(int(train_task_id))
                app_id = task_info["meta"]["app"]["id"]
                train_session = f'- **Training dashboard**:  <a href="/apps/{app_id}/sessions/{train_task_id}" target="_blank">open</a>'

            images_count = train_info.get("images_count")
            train_images_ids = train_info.get("train_images_ids")
            note_about_val_dataset = (
                f", {len(train_images_ids)} images in train, {images_count} images in validation"
            )
        else:
            gt_project_id = self._loader.gt_project_info.id
            gt_images_ids = self._loader._benchmark.gt_images_ids
            gt_dataset_ids = self._loader._benchmark.gt_dataset_ids
            if gt_images_ids is not None:
                note_about_val_dataset = (
                    f". Evaluated using validation subset - {len(gt_images_ids)} images"
                )
            elif gt_dataset_ids is not None:
                links = []
                for gt_dataset_id in gt_dataset_ids:
                    gt_dataset_name = self._loader._api.dataset.get_info_by_id(gt_dataset_id).name
                    link = f'<a href="/projects/{gt_project_id}/datasets/{gt_dataset_id}" target="_blank">{gt_dataset_name}</a>'
                    links.append(link)
                note_about_val_dataset = (
                    f". Evaluated on the dataset{'s' if len(links) > 1 else ''}: {', '.join(links)}"
                )
        return classes_str, note_about_val_dataset, train_session
