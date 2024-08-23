from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer

from supervisely.project.project_meta import ProjectMeta


class ExplorerGrid(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.clickable = True
        self.has_diffs_view = True

        optimal_conf = round(self.f1_optimal_conf, 1)
        filters = [{"confidence": [optimal_conf, 1]}]
        self.schema = Schema(
            markdown_explorer=Widget.Markdown(title="Explore Predictions", is_header=True),
            gallery=Widget.Gallery(filters=filters),
        )

    def _get_gallery(self, widget: Widget.Gallery, limit: Optional[int] = None) -> dict:
        res = {}
        api = self._loader._api
        pred_project_id = self._loader.dt_project_info.id
        pred_dataset = api.dataset.get_list(pred_project_id)[0]
        project_meta = ProjectMeta.from_json(api.project.get_meta(pred_project_id))
        pred_image_infos = api.image.get_list(dataset_id=pred_dataset.id, limit=limit)
        pred_image_ids = [x.id for x in pred_image_infos]
        ann_infos = api.annotation.download_batch(pred_dataset.id, pred_image_ids)

        for idx, (pred_image, ann_info) in enumerate(zip(pred_image_infos, ann_infos)):
            image_name = pred_image.name
            image_url = pred_image.full_storage_url
            widget.gallery.append(
                title=image_name,
                image_url=image_url,
                annotation_info=ann_info,
                column_index=idx % 3,
                project_meta=project_meta,
                ignore_tags_filtering=["outcome"],
            )
        res.update(widget.gallery.get_json_state())
        res.update(widget.gallery.get_json_data()["content"])
        res["layoutData"] = res.pop("annotations")
        res["projectMeta"] = project_meta.to_json()

        return res

    def get_gallery(self, widget: Widget.Gallery):
        return self._get_gallery(widget, limit=8)

    def get_gallery_click_data(self, widget: Widget.Gallery):
        res = {}

        res["layoutTemplate"] = [{"skipObjectTagsFiltering": ["outcome"]}] * 3
        click_data = res.setdefault("clickData", {})
        explore = click_data.setdefault("explore", {})
        explore["title"] = "Explore all predictions"
        images_ids = explore.setdefault("imagesIds", [])

        images = list(self._loader.dt_images_dct.values())
        images_ids.extend([x.id for x in images])

        return res

    def get_diff_gallery_data(self, widget: Widget.Gallery) -> Optional[dict]:
        res = {}

        res["layoutTemplate"] = [
            {"skipObjectTagsFiltering": True, "columnTitle": "Ground Truth"},
            {"skipObjectTagsFiltering": ["outcome"], "columnTitle": "Prediction"},
            {"skipObjectTagsFiltering": ["confidence"], "columnTitle": "Difference"},
        ]

        click_data = res.setdefault("clickData", {})

        l1 = list(self._loader.gt_images_dct.values())
        l2 = list(self._loader.dt_images_dct.values())
        l3 = list(self._loader.diff_images_dct.values())

        pred_anns = self._loader.dt_ann_jsons  # {image_id: ann_json}
        diff_anns = self._loader.diff_ann_jsons  # {image_id: ann_json}

        optimal_conf = round(self.f1_optimal_conf, 1)
        default_filters = [
            {"type": "tag", "tagId": "confidence", "value": [optimal_conf, 1]},
            # {"type": "tag", "tagId": "outcome", "value": "FP"},
        ]
        for gt, pred, diff, pred_ann, diff_ann in zip(
            l1, l2, l3, pred_anns.items(), diff_anns.items()
        ):
            assert gt.name == pred.name == diff.name
            key = click_data.setdefault(str(pred.id), {})
            key["imagesIds"] = [gt.id, pred.id, diff.id]
            key["filters"] = default_filters
            key["title"] = f"Image: {pred.name}"
            image_id, ann_json = pred_ann
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

            image_id, ann_json = diff_ann
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

    # def get_gallery_modal(self, widget: Widget.Gallery):
    #     res = self.get_gallery(widget)

    #     res.pop("layout")
    #     res.pop("layoutData")

    #     return res
