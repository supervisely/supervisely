import datetime
from pathlib import Path
from typing import Optional

from supervisely.api.module_api import ApiField
from supervisely.nn.benchmark.comparison.semantic_segmentation.vis_metrics import (
    Overview,
)
from supervisely.nn.benchmark.visualization.renderer import Renderer
from supervisely.nn.benchmark.visualization.widgets import (
    ContainerWidget,
    GalleryWidget,
    MarkdownWidget,
)
from supervisely.project.project_meta import ProjectMeta


class BaseComparisonVisualizer:
    vis_texts = None
    ann_opacity = None
    report_name = "Model Comparison Report.lnk"

    def __init__(self, comparison):
        self.comparison = comparison
        self.api = comparison.api
        self.eval_results = comparison.eval_results
        self.gt_project_info = None
        self.gt_project_meta = None
        # self._widgets_created = False

        for eval_result in self.eval_results:
            eval_result.api = self.api  # add api to eval_result for overview widget
            self._get_eval_project_infos(eval_result)

        self._create_widgets()
        layout = self._create_layout()

        self.renderer = Renderer(
            layout,
            str(Path(self.comparison.workdir, "visualizations")),
            report_name=self.report_name,
        )

    def visualize(self):
        return self.renderer.visualize()

    def upload_results(self, team_id: int, remote_dir: str, progress=None):
        return self.renderer.upload_results(self.api, team_id, remote_dir, progress)

    def _create_widgets(self):
        raise NotImplementedError("Have to implement in subclasses")

    def _create_layout(self):
        raise NotImplementedError("Have to implement in subclasses")

    def _create_header(self) -> MarkdownWidget:
        """Creates header widget"""
        me = self.api.user.get_my_info().login
        current_date = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        header_main_text = " ∣ ".join(  #  vs. or | or ∣
            eval_res.name for eval_res in self.comparison.eval_results
        )
        header_text = self.vis_texts.markdown_header.format(header_main_text, me, current_date)
        header = MarkdownWidget("markdown_header", "Header", text=header_text)
        return header

    def _create_overviews(self, vm: Overview, grid_cols: Optional[int] = None) -> ContainerWidget:
        """Creates overview widgets"""
        overview_widgets = vm.overview_widgets
        if grid_cols is None:
            grid_cols = 2
            if len(overview_widgets) > 2:
                grid_cols = 3
            if len(overview_widgets) % 4 == 0:
                grid_cols = 4
        return ContainerWidget(
            overview_widgets,
            name="overview_container",
            title="Overview",
            grid=True,
            grid_cols=grid_cols,
        )

    def _create_explore_modal_table(
        self, columns_number=3, click_gallery_id=None, hover_text=None
    ) -> GalleryWidget:
        gallery = GalleryWidget(
            "all_predictions_modal_gallery",
            is_modal=True,
            columns_number=columns_number,
            click_gallery_id=click_gallery_id,
            opacity=self.ann_opacity,
        )
        gallery.set_project_meta(self.eval_results[0].pred_project_meta)
        if hover_text:
            gallery.add_image_left_header(hover_text)
        return gallery

    def _create_diff_modal_table(self, columns_number=3) -> GalleryWidget:
        gallery = GalleryWidget(
            "diff_predictions_modal_gallery",
            is_modal=True,
            columns_number=columns_number,
            opacity=self.ann_opacity,
        )
        gallery.set_project_meta(self.eval_results[0].pred_project_meta)
        return gallery

    def _create_clickable_label(self):
        return MarkdownWidget("clickable_label", "", text=self.vis_texts.clickable_label)

    def _get_eval_project_infos(self, eval_result):
        # get project infos
        if self.gt_project_info is None:
            self.gt_project_info = self.api.project.get_info_by_id(eval_result.gt_project_id)
        eval_result.gt_project_info = self.gt_project_info
        eval_result.pred_project_info = self.api.project.get_info_by_id(eval_result.pred_project_id)

        # get project metas
        if self.gt_project_meta is None:
            self.gt_project_meta = ProjectMeta.from_json(
                self.api.project.get_meta(eval_result.gt_project_id)
            )
        eval_result.gt_project_meta = self.gt_project_meta
        eval_result.pred_project_meta = ProjectMeta.from_json(
            self.api.project.get_meta(eval_result.pred_project_id)
        )

        # get dataset infos
        filters = None
        if eval_result.gt_dataset_ids is not None:
            filters = [
                {
                    ApiField.FIELD: ApiField.ID,
                    ApiField.OPERATOR: "in",
                    ApiField.VALUE: eval_result.gt_dataset_ids,
                }
            ]
        eval_result.gt_dataset_infos = self.api.dataset.get_list(
            eval_result.gt_project_id,
            filters=filters,
            recursive=True,
        )

        # eval_result.pred_dataset_infos = self.api.dataset.get_list(
        #     eval_result.pred_project_id, recursive=True
        # )
