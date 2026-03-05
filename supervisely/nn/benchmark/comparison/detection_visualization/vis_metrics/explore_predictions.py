from typing import List, Tuple

from supervisely.annotation.annotation import Annotation
from supervisely.api.image_api import ImageInfo
from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.widgets import GalleryWidget, MarkdownWidget
from supervisely.project.project_meta import ProjectMeta


class ExplorePredictions(BaseVisMetrics):
    """Gallery and widgets to explore prediction differences across compared detection models."""

    MARKDOWN_DIFFERENCE = "markdown_explore_difference"
    GALLERY_DIFFERENCE = "explore_difference_gallery"
    MARKDOWN_SAME_ERRORS = "markdown_explore_same_errors"
    GALLERY_SAME_ERRORS = "explore_same_error_gallery"

    @property
    def difference_predictions_md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_explore_difference
        return MarkdownWidget(self.MARKDOWN_DIFFERENCE, "Explore Predictions", text)

    @property
    def explore_gallery(self) -> GalleryWidget:
        columns_number = len(self.eval_results) + 1
        *data, min_conf = self._get_sample_data()
        default_filters = [{"confidence": [min_conf, 1]}]
        gallery = GalleryWidget(
            self.GALLERY_DIFFERENCE, columns_number=columns_number, filters=default_filters
        )
        gallery.add_image_left_header("Click to explore more")
        gallery.set_project_meta(self.eval_results[0].gt_project_meta)
        gallery.set_images(*data)
        click_data = self.get_click_data_explore_all()
        gallery.set_click_data(self.explore_modal_table.id, click_data)
        gallery.set_show_all_data(self.explore_modal_table.id, click_data)
        gallery._gallery._update_filters()

        return gallery

    def _get_sample_data(self) -> Tuple[List[ImageInfo], List[Annotation], List[ProjectMeta]]:
        images = []
        annotations = []
        metas = [self.eval_results[0].gt_project_meta]
        skip_tags_filtering = [True]
        min_conf = float("inf")
        sample_names = self._get_evaluated_image_names(limit=5)
        if len(sample_names) == 0:
            raise RuntimeError("Failed to prepare sample gallery: no evaluated images were found.")

        gt_eval_res = self.eval_results[0]
        gt_map = self._get_project_images_by_names(gt_eval_res.gt_project_id, sample_names)
        column_maps = [gt_map]

        for eval_res in self.eval_results:
            metas.append(eval_res.pred_project_meta)
            pred_map = self._get_project_images_by_names(eval_res.pred_project_id, sample_names)
            column_maps.append(pred_map)
            skip_tags_filtering.append(False)
            min_conf = min(min_conf, eval_res.mp.conf_threshold)

        common_names = [name for name in sample_names if all(name in col_map for col_map in column_maps)]
        if len(common_names) == 0:
            raise RuntimeError(
                "Failed to prepare sample gallery: no common evaluated images were found "
                "across GT and prediction projects."
            )

        for name in common_names:
            for col_map in column_maps:
                image_info, ann_info = col_map[name]
                images.append(image_info)
                annotations.append(ann_info)
        return images, annotations, metas, skip_tags_filtering, min_conf

    def _get_evaluated_image_names(self, limit: int = None) -> List[str]:
        names = sorted({img["file_name"] for img in self.eval_results[0].coco_gt.imgs.values()})
        if limit is not None:
            return names[:limit]
        return names

    def _get_project_images_by_names(self, project_id: int, image_names: List[str]):
        api = self.eval_results[0].api
        image_names_set = set(image_names)
        result = {}

        for ds in api.dataset.get_list(project_id, recursive=True):
            image_infos = api.image.get_list(ds.id, force_metadata_for_links=False)
            image_infos = [info for info in image_infos if info.name in image_names_set]
            if len(image_infos) == 0:
                continue

            image_ids = [info.id for info in image_infos]
            ann_infos = api.annotation.download_batch(
                ds.id, image_ids, force_metadata_for_links=False
            )
            ann_infos_by_id = {ann.image_id: ann for ann in ann_infos}

            for image_info in image_infos:
                ann_info = ann_infos_by_id.get(image_info.id)
                if ann_info is not None and image_info.name not in result:
                    result[image_info.name] = (image_info, ann_info)
        return result

    def get_click_data_explore_all(self) -> dict:
        res = {}

        res["projectMeta"] = self.eval_results[0].gt_project_meta.to_json()
        res["layoutTemplate"] = [None, None, None]

        res["layoutTemplate"] = [{"skipObjectTagsFiltering": True, "columnTitle": "Ground Truth"}]
        # for i in range(len(self.eval_results)):
        for idx, eval_res in enumerate(self.eval_results, 1):
            res["layoutTemplate"].append({"columnTitle": f"[{idx}] {eval_res.name}"})

        click_data = res.setdefault("clickData", {})
        explore = click_data.setdefault("explore", {})
        explore["title"] = "Explore all predictions"

        images_ids = []
        min_conf = float("inf")
        all_names = self._get_evaluated_image_names()

        gt_eval_res = self.eval_results[0]
        gt_map = self._get_project_images_by_names(gt_eval_res.gt_project_id, all_names)
        pred_maps = []
        for eval_res in self.eval_results:
            pred_maps.append(self._get_project_images_by_names(eval_res.pred_project_id, all_names))
            min_conf = min(min_conf, eval_res.mp.conf_threshold)

        column_maps = [gt_map] + pred_maps
        common_names = [name for name in all_names if all(name in col_map for col_map in column_maps)]
        if len(common_names) == 0:
            raise RuntimeError(
                "Failed to build explore data: no common evaluated images were found "
                "across GT and prediction projects."
            )

        for col_map in column_maps:
            images_ids.append([col_map[name][0].id for name in common_names])

        explore["imagesIds"] = list(i for x in zip(*images_ids) for i in x)
        explore["filters"] = [{"type": "tag", "tagId": "confidence", "value": [min_conf, 1]}]

        return res
