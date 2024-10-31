from typing import List, Tuple

from supervisely.annotation.annotation import Annotation
from supervisely.api.image_api import ImageInfo
from supervisely.nn.benchmark.comparison.detection_visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import GalleryWidget, MarkdownWidget
from supervisely.project.project_meta import ProjectMeta


class ExplorePredictions(BaseVisMetric):

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
        gallery.show_all_button = True
        gallery.set_project_meta(self.eval_results[0].gt_project_meta)
        gallery.set_images(*data)
        gallery.add_on_click(
            self.explore_modal_table.id, self.get_click_data_explore_all(), columns_number * 3
        )
        gallery._gallery._filters
        gallery._gallery._update_filters()

        return gallery

    def _get_sample_data(self) -> Tuple[List[ImageInfo], List[Annotation], List[ProjectMeta]]:
        images = []
        annotations = []
        metas = [self.eval_results[0].gt_project_meta]
        skip_tags_filtering = []
        api = self.eval_results[0].api
        min_conf = float("inf")
        for idx, eval_res in enumerate(self.eval_results):
            if idx == 0:
                dataset_info = api.dataset.get_list(eval_res.gt_project_id)[0]
                image_infos = api.image.get_list(dataset_info.id, limit=5)
                images_ids = [image_info.id for image_info in image_infos]
                images.append(image_infos)
                anns = api.annotation.download_batch(dataset_info.id, images_ids)
                annotations.append(anns)
                skip_tags_filtering.append(True)
            metas.append(eval_res.dt_project_meta)
            dataset_info = api.dataset.get_list(eval_res.dt_project_id)[0]
            image_infos = eval_res.api.image.get_list(dataset_info.id, limit=5)
            images_ids = [image_info.id for image_info in image_infos]
            images.append(image_infos)
            anns = eval_res.api.annotation.download_batch(dataset_info.id, images_ids)
            annotations.append(anns)
            skip_tags_filtering.append(False)
            min_conf = min(min_conf, eval_res.f1_optimal_conf)

        images = list(i for x in zip(*images) for i in x)
        annotations = list(i for x in zip(*annotations) for i in x)
        return images, annotations, metas, skip_tags_filtering, min_conf

    def get_click_data_explore_all(self) -> dict:
        res = {}

        res["projectMeta"] = self.eval_results[0].gt_project_meta.to_json()
        res["layoutTemplate"] = [None, None, None]

        res["layoutTemplate"] = [{"skipObjectTagsFiltering": True, "columnTitle": "Ground Truth"}]
        for i in range(len(self.eval_results)):
            res["layoutTemplate"].append({"columnTitle": f"Model {i + 1}"})

        click_data = res.setdefault("clickData", {})
        explore = click_data.setdefault("explore", {})
        explore["title"] = "Explore all predictions"

        images_ids = []
        api = self.eval_results[0].api
        min_conf = float("inf")
        for idx, eval_res in enumerate(self.eval_results):
            if idx == 0:
                dataset_infos = api.dataset.get_list(eval_res.gt_project_id)
                current_images_ids = []
                for ds in dataset_infos:
                    image_infos = eval_res.api.image.get_list(ds.id)
                    current_images_ids.extend([image_info.id for image_info in image_infos])
                images_ids.append(current_images_ids)

            current_images_ids = []
            dataset_infos = api.dataset.get_list(eval_res.dt_project_id)
            for ds in dataset_infos:
                image_infos = eval_res.api.image.get_list(ds.id)
                current_images_ids.extend([image_info.id for image_info in image_infos])
            images_ids.append(current_images_ids)

            min_conf = min(min_conf, eval_res.f1_optimal_conf)

        explore["imagesIds"] = list(i for x in zip(*images_ids) for i in x)
        explore["filters"] = [{"type": "tag", "tagId": "confidence", "value": [min_conf, 1]}]

        return res
