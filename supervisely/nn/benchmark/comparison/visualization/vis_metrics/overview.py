from typing import List

from supervisely.nn.benchmark.comparison.evaluation_result import EvalResult
from supervisely.nn.benchmark.comparison.visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.comparison.visualization.widgets import (
    ChartWidget,
    MarkdownWidget,
    TableWidget,
)


class Overview(BaseVisMetric):

    MARKDOWN_OVERVIEW = "markdown_overview"
    CHART = "chart_key_metrics"

    def __init__(self, vis_texts, eval_results: List[EvalResult]) -> None:
        """
        Class to create widgets for the overview block
        overview_widgets property returns list of MarkdownWidget with information about the model
        chart_widget property returns ChartWidget with Scatterpolar chart of the base metrics with each
        evaluation result metrics displayed
        """
        super().__init__(vis_texts, eval_results)

    @property
    def overview_widgets(self) -> List[MarkdownWidget]:
        self.formats = []
        for eval_result in self.eval_results:

            url = eval_result.inference_info.get("checkpoint_url")
            link_text = eval_result.inference_info.get("custom_checkpoint_path")
            if link_text is None:
                link_text = url
            link_text = link_text.replace("_", "\_")

            # Note about validation dataset
            classes_str, note_about_val_dataset, train_session = self.get_overview_info(eval_result)

            checkpoint_name = eval_result.inference_info.get("deploy_params", {}).get(
                "checkpoint_name", ""
            )
            model_name = eval_result.inference_info.get("model_name") or "Custom"

            formats = [
                model_name.replace("_", "\_"),
                checkpoint_name.replace("_", "\_"),
                eval_result.inference_info.get("architecture"),
                eval_result.inference_info.get("task_type"),
                eval_result.inference_info.get("runtime"),
                url,
                link_text,
                eval_result.gt_project_info.id,
                eval_result.gt_project_info.name,
                classes_str,
                note_about_val_dataset,
                train_session,
                self.vis_texts.docs_url,
            ]
            self.formats.append(formats)

        text_template: str = getattr(self.vis_texts, self.MARKDOWN_OVERVIEW)
        return [
            MarkdownWidget(
                name=self.MARKDOWN_OVERVIEW, title="Overview", text=text_template.format(*formats)
            )
            for formats in self.formats
        ]

    @property
    def table_widget(self) -> TableWidget:
        res = {}

        columns = ["metrics"] + [
            f"[{i+1}] {eval_result.name}" for i, eval_result in enumerate(self.eval_results)
        ]

        all_metrics = [eval_result.mp.base_metrics() for eval_result in self.eval_results]
        res["content"] = []

        for metric in all_metrics[0].keys():
            row = [metric] + [round(metrics[metric], 2) for metrics in all_metrics]
            dct = {
                "row": row,
                "id": metric,
                "items": row,
            }
            res["content"].append(dct)

        columns_options = [{"disableSort": True} for _ in columns]

        res["columns"] = columns
        res["columnsOptions"] = columns_options

        return TableWidget(
            name="table_key_metrics",
            data=res,
            show_header_controls=False,
            fix_columns=1,
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
        gt_images_ids = eval_result.gt_images_ids
        train_info = eval_result.train_info
        if gt_images_ids is not None:
            val_imgs_cnt = len(gt_images_ids)
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

        if gt_images_ids is not None:
            images_str += f". Evaluated using subset - {val_imgs_cnt} images"
        elif gt_dataset_ids is not None:
            links = [
                f'<a href="/projects/{gt_project_id}/datasets/{ds.id}" target="_blank">{ds.name}</a>'
                for ds in datasets
            ]
            images_str += (
                f". Evaluated on the dataset{'s' if len(links) > 1 else ''}: {', '.join(links)}"
            )
        else:
            images_str += f". Evaluated on the whole project ({val_imgs_cnt} images)"

        return classes_str, images_str, train_session

    def get_figure(self):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error

        # Overall Metrics
        fig = go.Figure()
        for i, eval_result in enumerate(self.eval_results):
            name = f"[{i + 1}] {eval_result.name}"
            base_metrics = eval_result.mp.base_metrics()
            r = list(base_metrics.values())
            theta = [eval_result.mp.metric_names[k] for k in base_metrics.keys()]
            fig.add_trace(
                go.Scatterpolar(
                    r=r + [r[0]],
                    theta=theta + [theta[0]],
                    fill="toself",
                    name=name,
                    hovertemplate=name + "<br>%{theta}: %{r:.2f}<extra></extra>",
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
