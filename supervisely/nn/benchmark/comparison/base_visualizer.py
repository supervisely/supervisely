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
from supervisely.nn.benchmark.visualization.widgets.notification.notification import (
    NotificationWidget,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


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
        self._warning_notification = self._create_warning_notification_if_needed()

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
        if getattr(eval_result, "gt_project_info", None) is None:
            project_info = self.api.project.get_info_by_id(eval_result.gt_project_id)
            # if project_info is None:
            #     logger.warning(
            #         "Ground truth project with ID %s not found.", eval_result.gt_project_id
            #     )
            self.gt_project_info = project_info or eval_result.project_info
            eval_result.gt_project_info = self.gt_project_info

        if getattr(eval_result, "pred_project_info", None) is None:
            try:
                project_info = self.api.project.get_info_by_id(eval_result.pred_project_id)
                if project_info is None:
                    logger.warning(
                        "Prediction project with ID %s not found.", eval_result.pred_project_id
                    )
                else:
                    eval_result.pred_project_info = project_info
            except Exception as e:
                logger.warning("Error retrieving prediction project info: %s", e)

        # get project metas
        if getattr(eval_result, "gt_project_meta", None) is None:
            project_meta_json = self.api.project.get_meta(eval_result.gt_project_id)
            # if project_meta_json is None:
            #     logger.warning(
            #         "Ground truth project meta for project ID %s not found.",
            #         eval_result.gt_project_id,
            #     )
            self.gt_project_meta = ProjectMeta.from_json(project_meta_json)
            eval_result.gt_project_meta = self.gt_project_meta or eval_result.project_meta

        if getattr(eval_result, "pred_project_meta", None) is None:
            pred_project_meta_json = self.api.project.get_meta(eval_result.pred_project_id)
            if pred_project_meta_json is None:
                logger.warning(
                    "Prediction project meta for project ID %s not found.",
                    eval_result.pred_project_id,
                )
            else:
                eval_result.pred_project_meta = ProjectMeta.from_json(pred_project_meta_json)

        # get dataset infos
        if getattr(eval_result, "gt_dataset_infos", None) is None:
            filters = None
            if eval_result.gt_dataset_ids is not None:
                filters = [
                    {
                        ApiField.FIELD: ApiField.ID,
                        ApiField.OPERATOR: "in",
                        ApiField.VALUE: eval_result.gt_dataset_ids,
                    }
                ]
            try:
                eval_result.gt_dataset_infos = self.api.dataset.get_list(
                    eval_result.gt_project_id,
                    filters=filters,
                    recursive=True,
                )
            except Exception as e:
                logger.warning("Error retrieving ground truth dataset infos: %s", e)
                eval_result.gt_dataset_infos = None
            if eval_result.gt_dataset_infos is None:
                eval_result.gt_dataset_infos = eval_result.dataset_infos

    # @TODO: evaluate whether project existance notification is needed
    def _create_warning_notification_if_needed(self):
        NOTIFICATION = "overlap_notification"
        images_overlap_partial = bool(getattr(self.comparison, "images_partially_matched", False))
        classes_overlap_partial = bool(getattr(self.comparison, "classes_partially_matched", False))

        # Get matched statistics if available
        matched_classes_dict = getattr(self.comparison, "matched_classes_dict", None)
        matched_images_dict = getattr(self.comparison, "matched_images_dict", None)

        classes_desc = ""
        if matched_classes_dict:
            classes_desc = (
                f"<b>{matched_classes_dict['current']} out of {matched_classes_dict['max']} classes "
                f"({matched_classes_dict['percentage']})</b> are included in the comparison."
            )

        images_desc = ""
        if matched_images_dict:
            images_desc = (
                f"<b>{matched_images_dict['current']} out of {matched_images_dict['max']} images "
                f"({matched_images_dict['percentage']})</b> are included in the comparison."
            )

        if images_overlap_partial and classes_overlap_partial:
            return NotificationWidget(
                name=NOTIFICATION,
                title="Warning: Images and classes only partially overlap across evaluations.",
                desc=(
                    "<br>The comparison includes only items present in every evaluation (set intersection). "
                    f"Images or classes missing in any evaluation are excluded. <br>{images_desc} <br>{classes_desc} <br>"
                    "Align datasets and class definitions across evaluations for accurate results."
                ),
            )
        elif images_overlap_partial:
            return NotificationWidget(
                name=NOTIFICATION,
                title="Warning: Images only partially overlap across evaluations.",
                desc=(
                    "<br>Only images present in every evaluation are compared. "
                    f"Images missing from any evaluation are excluded. <br>{images_desc} <br>"
                    "Align datasets to ensure a fair comparison."
                ),
            )
        elif classes_overlap_partial:
            return NotificationWidget(
                name=NOTIFICATION,
                title="Warning: Classes only partially overlap across evaluations.",
                desc=(
                    "<br>Only classes present in every evaluation are compared. "
                    f"Classes missing from any evaluation are excluded. <br>{classes_desc} <br>"
                    "Align class definitions across evaluations for accurate results."
                ),
            )
        return None
