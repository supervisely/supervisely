from typing import Dict

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import GalleryWidget, MarkdownWidget


class ExplorePredictions(DetectionVisMetric):
    MARKDOWN = "explore_predictions"
    GALLERY = "explore_predictions"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clickable = True

    @property
    def md(self) -> MarkdownWidget:
        conf_threshold_info = "Differences are calculated only for the optimal confidence threshold, allowing you to focus on the most accurate predictions made by the model."
        if self.eval_result.mp.custom_conf_threshold is not None:
            conf_threshold_info = (
                "Differences are calculated for the custom confidence threshold (set manually)."
            )
        text = self.vis_texts.markdown_explorer.format(conf_threshold_info)

        return MarkdownWidget(self.MARKDOWN, "Explore Predictions", text)

    def gallery(self, opacity) -> GalleryWidget:
        default_filters = [{"confidence": [self.eval_result.mp.conf_threshold, 1]}]
        gallery = GalleryWidget(
            self.GALLERY, columns_number=3, filters=default_filters, opacity=opacity
        )
        gallery.add_image_left_header("Compare with GT")
        gallery.set_project_meta(self.eval_result.filtered_project_meta)

        gallery.set_images(
            image_infos=self.eval_result.sample_images,
            ann_infos=self.eval_result.sample_anns,
        )
        gallery._gallery._update_filters()

        # set click data for diff gallery
        self.explore_modal_table.set_click_data(
            self.diff_modal_table.id,
            self.get_diff_data(),
            get_key="(payload) => `${payload.annotation.imageId}`",
        )

        # set click data for explore gallery
        gallery.set_click_data(
            self.diff_modal_table.id,
            self.get_diff_data(),
            get_key="(payload) => `${payload.annotation.image_id}`",
        )

        gallery.set_show_all_data(
            self.explore_modal_table.id,
            self.get_click_data(),
        )
        return gallery

    def get_click_data(self) -> dict:
        res = {}

        res["layoutTemplate"] = [{"skipObjectTagsFiltering": ["outcome"]}] * 3
        click_data = res.setdefault("clickData", {})
        explore = click_data.setdefault("explore", {})
        explore["filters"] = [
            {
                "type": "tag",
                "tagId": "confidence",
                "value": [self.eval_result.mp.conf_threshold, 1],
            }
        ]
        explore["title"] = "Explore all predictions"
        images_ids = explore.setdefault("imagesIds", [])

        images_ids.extend(
            [d.pred_image_info.id for d in self.eval_result.matched_pair_data.values()]
        )

        return res

    def get_diff_data(self) -> Dict:
        res = {}

        res["layoutTemplate"] = [
            {"skipObjectTagsFiltering": True, "columnTitle": "Ground Truth"},
            {"skipObjectTagsFiltering": ["outcome"], "columnTitle": "Prediction"},
            {"skipObjectTagsFiltering": ["confidence"], "columnTitle": "Difference"},
        ]

        click_data = res.setdefault("clickData", {})

        default_filters = [
            {
                "type": "tag",
                "tagId": "confidence",
                "value": [self.eval_result.mp.conf_threshold, 1],
            },
        ]
        for pairs_data in self.eval_result.matched_pair_data.values():
            gt = pairs_data.gt_image_info
            pred = pairs_data.pred_image_info
            diff = pairs_data.diff_image_info
            assert gt.name == pred.name == diff.name
            for img_id in [gt.id, pred.id, diff.id]:
                key = click_data.setdefault(str(img_id), {})
                key["imagesIds"] = [gt.id, pred.id, diff.id]
                key["filters"] = default_filters
                key["title"] = f"Image: {gt.name}"

                object_bindings = []
                for img in [pred, diff]:
                    if img == pred:
                        ann_json = pairs_data.pred_annotation.to_json()
                    else:
                        ann_json = pairs_data.diff_annotation.to_json()
                    for obj in ann_json["objects"]:
                        for tag in obj["tags"]:
                            if tag["name"] == "matched_gt_id":
                                object_bindings.append(
                                    [
                                        {
                                            "id": obj["id"],
                                            "annotationKey": img.id,
                                        },
                                        {
                                            "id": int(tag["value"]),
                                            "annotationKey": gt.id if img == pred else pred.id,
                                        },
                                    ]
                                )
                key["objectsBindings"] = object_bindings

        return res
