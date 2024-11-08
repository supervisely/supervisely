from typing import Dict

from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import GalleryWidget, MarkdownWidget


class ExplorePredictions(BaseVisMetric):

    def __init__(self, vis_texts, eval_result: ObjectDetectionEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result

    MARKDOWN = "explore_predictions"
    GALLERY = "explore_predictions"

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_explorer
        return MarkdownWidget(self.MARKDOWN, "Explore Predictions", text)

    @property
    def gallery(self) -> GalleryWidget:
        optimal_conf = self.eval_result.mp.f1_optimal_conf
        default_filters = [{"confidence": [optimal_conf, 1]}]
        gallery = GalleryWidget(self.GALLERY, columns_number=3, filters=default_filters)
        gallery.add_image_left_header("Click to explore more")
        gallery.show_all_button = True

        gallery.set_project_meta(self.eval_result.pred_project_meta)

        gallery.set_images(
            image_infos=self.eval_result.sample_images,
            ann_infos=self.eval_result.sample_anns,
        )
        # gallery.add_on_click(
        #     self.explore_modal_table.id, self.get_click_data_explore_all(), columns_number * 3
        # )
        gallery._gallery._update_filters()
        return gallery

    def get_gallery_click_data(self) -> dict:
        res = {}

        res["layoutTemplate"] = [{"skipObjectTagsFiltering": ["outcome"]}] * 3
        click_data = res.setdefault("clickData", {})
        explore = click_data.setdefault("explore", {})
        explore["title"] = "Explore all predictions"
        images_ids = explore.setdefault("imagesIds", [])

        images_ids.extend(
            [d.pred_image_info.id for d in self.eval_result.matched_pair_data.values()]
        )

        return res

    def get_gallery_diff_data(self) -> Dict:
        res = {}

        res["layoutTemplate"] = [
            {"skipObjectTagsFiltering": True, "columnTitle": "Ground Truth"},
            {"skipObjectTagsFiltering": ["outcome"], "columnTitle": "Prediction"},
            {"skipObjectTagsFiltering": ["confidence"], "columnTitle": "Difference"},
        ]

        click_data = res.setdefault("clickData", {})

        default_filters = [
            {"type": "tag", "tagId": "confidence", "value": [self.f1_optimal_conf, 1]},
            # {"type": "tag", "tagId": "outcome", "value": "FP"},
        ]
        for pairs_data in self.eval_result.matched_pair_data.values():
            gt = pairs_data.gt_image_info
            pred = pairs_data.pred_image_info
            diff = pairs_data.diff_image_info
            assert gt.name == pred.name == diff.name
            key = click_data.setdefault(str(pred.id), {})
            key["imagesIds"] = [gt.id, pred.id, diff.id]
            key["filters"] = default_filters
            key["title"] = f"Image: {pred.name}"
            image_id = pred.id
            ann_json = pairs_data.pred_annotation.to_json()
            assert image_id == pred.id
            object_bindings = []
            for obj in ann_json["objects"]:
                for tag in obj["tags"]:
                    if tag["name"] == "matched_gt_id":
                        object_bindings.append(
                            [
                                {
                                    "id": obj["id"],
                                    "annotationKey": image_id,
                                },
                                {
                                    "id": int(tag["value"]),
                                    "annotationKey": gt.id,
                                },
                            ]
                        )

            image_id = diff.id
            ann_json = pairs_data.diff_annotation.to_json()
            assert image_id == diff.id
            for obj in ann_json["objects"]:
                for tag in obj["tags"]:
                    if tag["name"] == "matched_gt_id":
                        object_bindings.append(
                            [
                                {
                                    "id": obj["id"],
                                    "annotationKey": image_id,
                                },
                                {
                                    "id": int(tag["value"]),
                                    "annotationKey": pred.id,
                                },
                            ]
                        )
            key["objectsBindings"] = object_bindings

        return res
