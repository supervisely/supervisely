from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import GalleryWidget, MarkdownWidget


class ExplorePredictions(SemanticSegmVisMetric):
    MARKDOWN = "explore_predictions"
    GALLERY = "explore_predictions"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clickable = True

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_explorer
        return MarkdownWidget(self.MARKDOWN, "Explore Predictions", text)

    def gallery(self, opacity) -> GalleryWidget:
        gallery = GalleryWidget(self.GALLERY, columns_number=3, opacity=opacity)
        gallery.set_project_meta(self.eval_result.filtered_project_meta)
        gallery.add_image_left_header("Click to explore more")

        gallery.set_images(
            image_infos=self.eval_result.sample_images,
            ann_infos=self.eval_result.sample_anns,
        )
        gallery._gallery._update_filters()

        # set click data for explore gallery
        gallery.set_click_data(self.explore_modal_table.id, self.get_click_data())
        gallery.set_show_all_data(self.explore_modal_table.id, self.get_click_data())
        return gallery

    def get_click_data(self) -> dict:
        res = {}

        res["layoutTemplate"] = [
            {"columnTitle": "Original Image"},
            {"columnTitle": "Ground Truth Masks"},
            {"columnTitle": "Predicted Masks"},
        ]
        click_data = res.setdefault("clickData", {})
        explore = click_data.setdefault("explore", {})
        explore["title"] = "Explore all predictions"
        images_ids = explore.setdefault("imagesIds", [])

        for data in self.eval_result.matched_pair_data.values():
            images_ids.extend(
                [data.diff_image_info.id, data.gt_image_info.id, data.pred_image_info.id]
            )

        return res
