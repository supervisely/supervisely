from ast import Dict
from typing import List, Tuple

from supervisely.annotation.annotation import Annotation
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.image_api import ImageInfo
from supervisely.api.module_api import ApiField
from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.evaluation_result import EvalResult
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
        api: Api = self.eval_results[0].api
        min_conf = float("inf")
        sample_images: List[ImageInfo] = [] # from 1st dt project
        sample_annotations: List[Annotation] = []
        sample_infos_by_dataset: Dict[str, List[ImageInfo]] = {}
        for idx, eval_res in enumerate(self.eval_results):
            metas.append(eval_res.pred_project_meta)
            min_conf = min(min_conf, eval_res.mp.conf_threshold)
            this_eval_res_images = []
            this_eval_res_annotations = []
            if idx == 0:
                for dataset_info in eval_res.pred_dataset_infos:    
                    batch_images = api.image.get_list(dataset_info.id, limit=5, force_metadata_for_links=False)
                    if len(batch_images) == 0:
                        continue

                    gt_dataset_info = api.dataset.get_info_by_name(eval_res.gt_project_id, name=dataset_info.name)
                    assert gt_dataset_info is not None, f"Dataset {dataset_info.name} not found in GT project"
                    
                    gt_batch_images = api.image.get_list(gt_dataset_info.id, filters=[{ApiField.FIELD: ApiField.NAME, ApiField.OPERATOR: "in", ApiField.VALUE: [info.name for info in batch_images]}], force_metadata_for_links=False)
                    assert len(gt_batch_images) == len(batch_images) is not None, "Failed to get GT images for gallery"
                    
                    sample_images.extend(gt_batch_images)
                    gt_batch_annotations = api.annotation.download_batch(gt_dataset_info.id, [info.id for info in gt_batch_images], force_metadata_for_links=False)
                    sample_annotations.extend(gt_batch_annotations)
                    batch_anns = api.annotation.download_batch(dataset_info.id, [info.id for info in batch_images], force_metadata_for_links=False)
                    this_eval_res_images.extend(batch_images)
                    this_eval_res_annotations.extend(batch_anns)

                    if len(sample_images) >= 5:
                        sample_infos_by_dataset[dataset_info.name] = gt_batch_images[:-(len(sample_images)-5)]
                        sample_images = sample_images[:5]
                        sample_annotations = sample_annotations[:5]
                        this_eval_res_images = this_eval_res_images[:5]
                        this_eval_res_annotations = this_eval_res_annotations[:5]
                        # gt
                        images.append(sample_images)
                        annotations.append(sample_annotations)
                        skip_tags_filtering.append(True)
                        break
                    else:
                        sample_infos_by_dataset[dataset_info.name] = gt_batch_images

                assert len(sample_images) > 0, "No images found in the DT project"
            else:
                for dataset_name, gt_image_infos in sample_infos_by_dataset:
                    dataset_info = api.dataset.get_info_by_name(eval_res.pred_project_id, dataset_name)
                    assert dataset_info is not None, f"Dataset {dataset_name} not found in DT project"
                    image_infos = eval_res.api.image.get_list(
                        dataset_info.id,
                        filters=[
                            {ApiField.FIELD: ApiField.NAME, ApiField.OPERATOR: "in", ApiField.VALUE: [info.name for info in gt_image_infos]}
                        ],
                        force_metadata_for_links=False,
                    )
                    anns = eval_res.api.annotation.download_batch(dataset_info.id, [info.id for info in image_infos], force_metadata_for_links=False)
                    this_eval_res_images.extend(image_infos)
                    this_eval_res_annotations.extend(anns)
                assert len(this_eval_res_images) == len(sample_images), "Failed to get DT images"

            images.append(this_eval_res_images)
            annotations.append(this_eval_res_annotations)
            skip_tags_filtering.append(False)

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
