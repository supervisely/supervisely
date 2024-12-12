from typing import List

from supervisely._utils import abs_url
from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.evaluation_result import EvalResult
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    MarkdownWidget,
    TableWidget,
)


class Overview(BaseVisMetrics):

    MARKDOWN_OVERVIEW = "markdown_overview"
    MARKDOWN_OVERVIEW_INFO = "markdown_overview_info"
    MARKDOWN_COMMON_OVERVIEW = "markdown_common_overview"
    CHART = "chart_key_metrics"

    def __init__(self, vis_texts, eval_results: List[EvalResult]) -> None:
        super().__init__(vis_texts, eval_results)

    @property
    def overview_md(self) -> List[MarkdownWidget]:
        info = []
        model_names = []
        for eval_result in self.eval_results:
            model_name = eval_result.name or "Custom"
            model_name = model_name.replace("_", "\_")
            model_names.append(model_name)

            info.append(
                [
                    eval_result.gt_project_info.id,
                    eval_result.gt_project_info.name,
                    eval_result.inference_info.get("task_type"),
                ]
            )
        if all([model_name == "Custom" for model_name in model_names]):
            model_name = "Custom models"
        elif all([model_name == model_names[0] for model_name in model_names]):
            model_name = model_names[0]
        else:
            model_name = " vs. ".join(model_names)

        info = [model_name] + info[0]

        text_template: str = getattr(self.vis_texts, self.MARKDOWN_COMMON_OVERVIEW)
        return MarkdownWidget(
            name=self.MARKDOWN_COMMON_OVERVIEW,
            title="Overview",
            text=text_template.format(*info),
        )

    @property
    def overview_widgets(self) -> List[MarkdownWidget]:
        all_formats = []
        for eval_result in self.eval_results:

            url = eval_result.inference_info.get("checkpoint_url")
            link_text = eval_result.inference_info.get("custom_checkpoint_path")
            if link_text is None:
                link_text = url
            link_text = link_text.replace("_", "\_")

            checkpoint_name = eval_result.checkpoint_name
            model_name = eval_result.inference_info.get("model_name") or "Custom"

            report = eval_result.api.file.get_info_by_path(self.team_id, eval_result.report_path)
            report_link = abs_url(f"/model-benchmark?id={report.id}")

            formats = [
                checkpoint_name,
                model_name.replace("_", "\_"),
                checkpoint_name.replace("_", "\_"),
                eval_result.inference_info.get("architecture"),
                eval_result.inference_info.get("runtime"),
                url,
                link_text,
                report_link,
            ]
            all_formats.append(formats)

        text_template: str = getattr(self.vis_texts, self.MARKDOWN_OVERVIEW_INFO)
        widgets = []
        for formats in all_formats:
            md = MarkdownWidget(
                name=self.MARKDOWN_OVERVIEW_INFO,
                title="Overview",
                text=text_template.format(*formats),
            )
            md.is_info_block = True
            widgets.append(md)
        return widgets

    def get_table_widget(self, latency, fps) -> TableWidget:
        res = {}

        columns = ["metrics"] + [f"[{i+1}] {r.name}" for i, r in enumerate(self.eval_results)]

        all_metrics = [eval_result.mp.key_metrics() for eval_result in self.eval_results]
        res["content"] = []

        for metric in all_metrics[0].keys():
            values = [m[metric] for m in all_metrics]
            values = [v if v is not None else "―" for v in values]
            values = [round(v, 2) if isinstance(v, float) else v for v in values]
            row = [metric] + values
            dct = {"row": row, "id": metric, "items": row}
            res["content"].append(dct)

        latency_row = ["Latency (ms)"] + latency
        res["content"].append({"row": latency_row, "id": latency_row[0], "items": latency_row})

        fps_row = ["FPS"] + fps
        res["content"].append({"row": fps_row, "id": fps_row[0], "items": fps_row})

        columns_options = [{"disableSort": True} for _ in columns]

        res["columns"] = columns
        res["columnsOptions"] = columns_options

        return TableWidget(
            name="table_key_metrics",
            data=res,
            show_header_controls=False,
            fix_columns=1,
            page_size=len(res["content"]),
        )

    @property
    def chart_widget(self) -> ChartWidget:
        return ChartWidget(name=self.CHART, figure=self.get_figure())

    def get_overview_info(self, eval_result: EvalResult):
        classes_cnt = len(eval_result.classes_whitelist)
        classes_str = "classes" if classes_cnt > 1 else "class"
        classes_str = f"{classes_cnt} {classes_str}"

        train_session, images_str = "", ""
        gt_project_id = eval_result.gt_project_info.id
        gt_dataset_ids = eval_result.gt_dataset_ids
        gt_images_cnt = eval_result.val_images_cnt
        train_info = eval_result.train_info
        total_imgs_cnt = eval_result.gt_project_info.items_count
        if gt_images_cnt is not None:
            val_imgs_cnt = gt_images_cnt
        elif gt_dataset_ids is not None:
            datasets = eval_result.gt_dataset_infos
            val_imgs_cnt = sum(ds.items_count for ds in datasets)
        else:
            val_imgs_cnt = eval_result.gt_project_info.items_count

        if train_info:
            train_task_id = train_info.get("app_session_id")
            if train_task_id:
                task_info = eval_result.api.task.get_info_by_id(int(train_task_id))
                app_id = task_info["meta"]["app"]["id"]
                train_session = f'- **Training dashboard**:  <a href="/apps/{app_id}/sessions/{train_task_id}" target="_blank">open</a>'

            train_imgs_cnt = train_info.get("images_count")
            images_str = f", {train_imgs_cnt} images in train, {val_imgs_cnt} images in validation"

        if gt_images_cnt is not None:
            images_str += (
                f", total {total_imgs_cnt} images. Evaluated using subset - {val_imgs_cnt} images"
            )
        elif gt_dataset_ids is not None:
            links = [
                f'<a href="/projects/{gt_project_id}/datasets/{ds.id}" target="_blank">{ds.name}</a>'
                for ds in datasets
            ]
            images_str += f", total {total_imgs_cnt} images. Evaluated on the dataset{'s' if len(links) > 1 else ''}: {', '.join(links)}"
        else:
            images_str += f", total {total_imgs_cnt} images. Evaluated on the whole project ({val_imgs_cnt} images)"

        return classes_str, images_str, train_session

    def get_figure(self):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error

        # Overall Metrics
        fig = go.Figure()
        for i, eval_result in enumerate(self.eval_results):
            name = f"[{i + 1}] {eval_result.name}"
            base_metrics = eval_result.mp.key_metrics().copy()
            base_metrics["mPixel accuracy"] = round(base_metrics["mPixel accuracy"] * 100, 2)
            r = list(base_metrics.values())
            theta = list(base_metrics.keys())
            fig.add_trace(
                go.Scatterpolar(
                    r=r + [r[0]],
                    theta=theta + [theta[0]],
                    name=name,
                    marker=dict(color=eval_result.color),
                    hovertemplate=name + "<br>%{theta}: %{r:.2f}<extra></extra>",
                )
            )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0, 105],
                    ticks="outside",
                ),
                angularaxis=dict(rotation=90, direction="clockwise"),
            ),
            dragmode=False,
            height=500,
            margin=dict(l=25, r=25, t=25, b=25),
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
            ),
        )
        return fig
