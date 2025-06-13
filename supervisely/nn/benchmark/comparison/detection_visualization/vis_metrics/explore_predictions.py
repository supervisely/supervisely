from typing import List, Tuple

from supervisely.annotation.annotation import Annotation
from supervisely.api.image_api import ImageInfo
from supervisely.api.module_api import ApiField
from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.widgets import GalleryWidget, MarkdownWidget
from supervisely.project.project_meta import ProjectMeta


class ExplorePredictions(BaseVisMetrics):

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
        skip_tags_filtering = []
        api = self.eval_results[0].api
        min_conf = float("inf")
        names = None
        ds_name = None
        for idx, eval_res in enumerate(self.eval_results):
            if idx == 0:
                dataset_info = eval_res.gt_dataset_infos[0]
                image_infos = api.image.get_list(dataset_info.id, limit=5, force_metadata_for_links=False)
                ds_name = dataset_info.name
                images_ids = [image_info.id for image_info in image_infos]
                names = [image_info.name for image_info in image_infos]
                images.append(image_infos)
                anns = api.annotation.download_batch(dataset_info.id, images_ids, force_metadata_for_links=False)
                annotations.append(anns)
                skip_tags_filtering.append(True)
            metas.append(eval_res.pred_project_meta)
            assert ds_name is not None, "Failed to get GT dataset name for gallery"

            dataset_info = api.dataset.get_info_by_name(eval_res.pred_project_id, ds_name)

            assert names is not None, "Failed to get GT image names for gallery"
            image_infos = eval_res.api.image.get_list(
                dataset_info.id,
                filters=[
                    {ApiField.FIELD: ApiField.NAME, ApiField.OPERATOR: "in", ApiField.VALUE: names}
                ],
                force_metadata_for_links=False,
            )
            images_ids = [image_info.id for image_info in image_infos]
            images.append(image_infos)
            anns = eval_res.api.annotation.download_batch(dataset_info.id, images_ids, force_metadata_for_links=False)
            annotations.append(anns)
            skip_tags_filtering.append(False)
            min_conf = min(min_conf, eval_res.mp.conf_threshold)

        images = list(i for x in zip(*images) for i in x)
        annotations = list(i for x in zip(*annotations) for i in x)
        return images, annotations, metas, skip_tags_filtering, min_conf

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
        api = self.eval_results[0].api
        min_conf = float("inf")
        names = None
        ds_names = None
        for idx, eval_res in enumerate(self.eval_results):
            if idx == 0:
                dataset_infos = eval_res.gt_dataset_infos
                ds_names = [ds.name for ds in dataset_infos]
                current_images_ids = []
                current_images_names = []
                for ds in dataset_infos:
                    image_infos = api.image.get_list(ds.id, force_metadata_for_links=False)
                    image_infos = sorted(image_infos, key=lambda x: x.name)
                    current_images_names.extend([image_info.name for image_info in image_infos])
                    current_images_ids.extend([image_info.id for image_info in image_infos])
                images_ids.append(current_images_ids)
                names = current_images_names

            dataset_infos = api.dataset.get_list(eval_res.pred_project_id)
            dataset_infos = [ds for ds in dataset_infos if ds.name in ds_names]
            dataset_infos = sorted(dataset_infos, key=lambda x: ds_names.index(x.name))
            current_images_infos = []
            for ds in dataset_infos:
                image_infos = api.image.get_list(ds.id, force_metadata_for_links=False)
                image_infos = [image_info for image_info in image_infos if image_info.name in names]
                current_images_infos.extend(image_infos)
            current_images_infos = sorted(current_images_infos, key=lambda x: names.index(x.name))
            images_ids.append([image_info.id for image_info in current_images_infos])

            min_conf = min(min_conf, eval_res.mp.conf_threshold)

        explore["imagesIds"] = list(i for x in zip(*images_ids) for i in x)
        explore["filters"] = [{"type": "tag", "tagId": "confidence", "value": [min_conf, 1]}]

        return res
