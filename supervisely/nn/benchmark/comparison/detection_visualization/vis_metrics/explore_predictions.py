from typing import List, Tuple

from supervisely.annotation.annotation import Annotation
from supervisely.api.image_api import ImageInfo
from supervisely.api.module_api import ApiField
from supervisely.sly_logger import logger
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
            self.GALLERY_DIFFERENCE,
            columns_number=columns_number,
            filters=default_filters,
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
        sample_keys = self._get_evaluated_image_keys(limit=5)
        if len(sample_keys) == 0:
            raise RuntimeError("Failed to prepare sample gallery: no evaluated images were found.")

        gt_eval_res = self.eval_results[0]
        gt_map = self._get_project_items_by_keys(gt_eval_res.gt_project_id, sample_keys)
        column_maps = [gt_map]

        for eval_res in self.eval_results:
            metas.append(eval_res.pred_project_meta)
            pred_map = self._get_project_items_by_keys(eval_res.pred_project_id, sample_keys)
            column_maps.append(pred_map)
            skip_tags_filtering.append(False)
            min_conf = min(min_conf, eval_res.mp.conf_threshold)

        common_keys = [key for key in sample_keys if all(key in col_map for col_map in column_maps)]
        if len(common_keys) == 0:
            raise RuntimeError(
                "Failed to prepare sample gallery: no common evaluated images were found "
                "across GT and prediction projects."
            )

        for key in common_keys:
            for col_map in column_maps:
                image_info, ann_info = col_map[key]
                images.append(image_info)
                annotations.append(ann_info)
        return images, annotations, metas, skip_tags_filtering, min_conf

    def _get_evaluated_image_keys(self, limit: int = None) -> List[Tuple[str, str]]:
        """Returns evaluated image keys as (dataset_name, image_name) pairs"""
        keys = sorted(
            {
                (img.get("dataset"), img.get("file_name"))
                for img in self.eval_results[0].coco_gt.imgs.values()
                if img.get("dataset") is not None and img.get("file_name") is not None
            }
        )
        if limit is not None:
            return keys[:limit]
        return keys

    def _get_project_items_by_keys(self, project_id: int, keys: List[Tuple[str, str]]):
        """
        Build mapping (dataset_name, image_name) -> (ImageInfo, AnnotationInfo) for given project.
        """
        api = self.eval_results[0].api
        result = {}

        ds_to_names = {}
        for ds_name, img_name in keys:
            ds_to_names.setdefault(ds_name, []).append(img_name)

        # Build dataset path->info map once per call, no caching on self.
        datasets = api.dataset.get_list(project_id, recursive=True)
        by_id = {ds.id: ds for ds in datasets}
        path_by_id = {}

        def get_path(ds):
            existing = path_by_id.get(ds.id)
            if existing is not None:
                return existing
            if ds.parent_id is None:
                path = ds.name
            else:
                parent = by_id.get(ds.parent_id)
                path = ds.name if parent is None else f"{get_path(parent)}/{ds.name}"
            path_by_id[ds.id] = path
            return path

        ds_path_map = {get_path(ds): ds for ds in datasets}
        for ds_name, img_names in ds_to_names.items():
            if len(img_names) == 0:
                continue
            ds_info = ds_path_map.get(ds_name)
            if ds_info is None:
                logger.warning(
                    f"Dataset '{ds_name}' not found in project {project_id}. Skipping.",
                    extra={"project_id": project_id, "dataset_name": ds_name},
                )
                continue

            infos = api.image.get_list(
                ds_info.id,
                filters=[
                    {ApiField.FIELD: ApiField.NAME, ApiField.OPERATOR: "in", ApiField.VALUE: img_names}
                ],
                force_metadata_for_links=False,
            )
            if len(infos) == 0:
                continue

            infos = sorted(infos, key=lambda x: img_names.index(x.name))
            image_ids = [info.id for info in infos]
            ann_infos = api.annotation.download_batch(
                ds_info.id, image_ids, force_metadata_for_links=False
            )
            ann_infos_by_id = {ann.image_id: ann for ann in ann_infos}

            for info in infos:
                ann = ann_infos_by_id.get(info.id)
                if ann is None:
                    continue
                key = (ds_name, info.name)
                if key not in result:
                    result[key] = (info, ann)
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
        all_keys = self._get_evaluated_image_keys()

        gt_eval_res = self.eval_results[0]
        gt_map = self._get_project_items_by_keys(gt_eval_res.gt_project_id, all_keys)
        pred_maps = []
        for eval_res in self.eval_results:
            pred_maps.append(self._get_project_items_by_keys(eval_res.pred_project_id, all_keys))
            min_conf = min(min_conf, eval_res.mp.conf_threshold)

        column_maps = [gt_map] + pred_maps
        common_keys = [key for key in all_keys if all(key in col_map for col_map in column_maps)]
        if len(common_keys) == 0:
            raise RuntimeError(
                "Failed to build explore data: no common evaluated images were found "
                "across GT and prediction projects."
            )

        for col_map in column_maps:
            images_ids.append([col_map[key][0].id for key in common_keys])

        explore["imagesIds"] = list(i for x in zip(*images_ids) for i in x)
        explore["filters"] = [{"type": "tag", "tagId": "confidence", "value": [min_conf, 1]}]

        return res
