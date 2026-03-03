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
                    batch_images = batch_images[:5 - len(sample_images)]
                    batch_images.sort(key=lambda x: x.name)
                    if len(batch_images) == 0:
                        continue

                    gt_dataset_info = api.dataset.get_info_by_name(eval_res.gt_project_id, name=dataset_info.name)
                    assert gt_dataset_info is not None, f"Dataset {dataset_info.name} not found in GT project"
                    
                    gt_batch_images = api.image.get_list(gt_dataset_info.id, filters=[{ApiField.FIELD: ApiField.NAME, ApiField.OPERATOR: "in", ApiField.VALUE: [info.name for info in batch_images]}], force_metadata_for_links=False)
                    gt_batch_images.sort(key=lambda x: x.name)
                    assert len(gt_batch_images) == len(batch_images) is not None, "Failed to get GT images for gallery"
                    

                    sample_images.extend(gt_batch_images)
                    gt_batch_annotations = api.annotation.download_batch(gt_dataset_info.id, [info.id for info in gt_batch_images], force_metadata_for_links=False)
                    sample_annotations.extend(gt_batch_annotations)
                    batch_anns = api.annotation.download_batch(dataset_info.id, [info.id for info in batch_images], force_metadata_for_links=False)
                    this_eval_res_images.extend(batch_images)
                    this_eval_res_annotations.extend(batch_anns)

                    if len(sample_images) >= 5:
                        sample_infos_by_dataset[dataset_info.name] = gt_batch_images[:5 - (len(sample_images) - len(gt_batch_images))]
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
                for dataset_name, gt_image_infos in sample_infos_by_dataset.items():
                    dataset_info = api.dataset.get_info_by_name(eval_res.pred_project_id, dataset_name)
                    assert dataset_info is not None, f"Dataset {dataset_name} not found in DT project"
                    image_infos = eval_res.api.image.get_list(
                        dataset_info.id,
                        filters=[
                            {ApiField.FIELD: ApiField.NAME, ApiField.OPERATOR: "in", ApiField.VALUE: [info.name for info in gt_image_infos]}
                        ],
                        force_metadata_for_links=False,
                    )
                    image_infos.sort(key=lambda x: x.name)
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
        api: Api = self.eval_results[0].api
        min_conf = float("inf")


        gt_images = {}
        for dataset_info in self.eval_results[0].gt_dataset_infos:
            ds_images = api.image.get_list(dataset_info.id, force_metadata_for_links=False)
            if len(ds_images) == 0:
                continue
            gt_images[dataset_info.name] = ds_images
        eval_images: List[Dict[str, List[ImageInfo]]] = []
        for eval_res in self.eval_results:
            min_conf = min(min_conf, eval_res.mp.conf_threshold)
            eval_images.append({})
            for dataset_info in eval_res.pred_dataset_infos:
                batch_images = api.image.get_list(dataset_info.id, force_metadata_for_links=False)
                if len(batch_images) == 0:
                    continue
                eval_images[-1][dataset_info.name] = batch_images
        for ds_name, gt_ds_images in gt_images.items():
            for gt_info in gt_ds_images:
                this_image_ids = [gt_info.id]
                for i in range(len(eval_images)):
                    this_eval_images = eval_images[i]
                    if ds_name not in this_eval_images:
                        break
                    for image in this_eval_images[ds_name]:
                        if image.name == gt_info.name:
                            this_image_ids.append(image.id)
                            break
                if len(this_image_ids) == len(self.eval_results) + 1:
                    images_ids.extend(this_image_ids)

        explore["imagesIds"] = images_ids
        explore["filters"] = [{"type": "tag", "tagId": "confidence", "value": [min_conf, 1]}]

        return res
