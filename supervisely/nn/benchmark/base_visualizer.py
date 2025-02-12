from typing import List, Tuple

from supervisely.annotation.annotation import Annotation
from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.api.module_api import ApiField
from supervisely.api.project_api import ProjectInfo
from supervisely.nn.benchmark.base_evaluator import BaseEvalResult
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.visualization.renderer import Renderer
from supervisely.nn.benchmark.visualization.widgets import GalleryWidget
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import tqdm_sly


class MatchedPairData:
    def __init__(
        self,
        gt_image_info: ImageInfo = None,
        pred_image_info: ImageInfo = None,
        diff_image_info: ImageInfo = None,
        gt_annotation: Annotation = None,
        pred_annotation: Annotation = None,
        diff_annotation: Annotation = None,
    ):
        self.gt_image_info = gt_image_info
        self.pred_image_info = pred_image_info
        self.diff_image_info = diff_image_info
        self.gt_annotation = gt_annotation
        self.pred_annotation = pred_annotation
        self.diff_annotation = diff_annotation


class BaseVisMetrics:

    def __init__(
        self,
        vis_texts,
        eval_results: List[BaseEvalResult],
        explore_modal_table: GalleryWidget = None,
        diff_modal_table: GalleryWidget = None,
    ) -> None:
        self.vis_texts = vis_texts
        self.eval_results = eval_results
        self.explore_modal_table = explore_modal_table
        self.diff_modal_table = diff_modal_table
        self.clickable = False


class BaseVisMetric(BaseVisMetrics):
    def __init__(
        self,
        vis_texts,
        eval_result: BaseEvalResult,
        explore_modal_table: GalleryWidget = None,
        diff_modal_table: GalleryWidget = None,
    ) -> None:
        super().__init__(vis_texts, [eval_result], explore_modal_table, diff_modal_table)
        self.eval_result = eval_result


class BaseVisualizer:
    cv_task = None
    report_name = "Model Evaluation Report.lnk"

    def __init__(
        self,
        api: Api,
        eval_results: List[BaseEvalResult],
        workdir="./visualizations",
        progress=None,
    ):
        self.api = api
        self.workdir = workdir
        self.eval_result = eval_results[0]  # for evaluation
        self.eval_results = eval_results  # for comparison

        self.renderer = None
        self.gt_project_info = None
        self.gt_project_meta = None
        self.gt_dataset_infos = None
        self.pbar = progress or tqdm_sly
        self.ann_opacity = 0.4

        with self.pbar(message="Fetching project and dataset infos", total=len(eval_results)) as p:
            for eval_result in self.eval_results:
                if eval_result.gt_project_id is not None:
                    self._get_eval_project_infos(eval_result)
                p.update(1)

    def _get_eval_project_infos(self, eval_result):
        # get project infos
        if self.gt_project_info is None:
            self.gt_project_info = self.api.project.get_info_by_id(eval_result.gt_project_id)
        eval_result.gt_project_info = self.gt_project_info
        eval_result.pred_project_info = self.api.project.get_info_by_id(eval_result.pred_project_id)

        # get project metas
        if self.gt_project_meta is None:
            self.gt_project_meta = ProjectMeta.from_json(
                self.api.project.get_meta(eval_result.gt_project_id)
            )
        eval_result.gt_project_meta = self.gt_project_meta
        eval_result.pred_project_meta = ProjectMeta.from_json(
            self.api.project.get_meta(eval_result.pred_project_id)
        )

        # get dataset infos
        filters = None
        if eval_result.gt_dataset_ids is not None:
            filters = [
                {
                    ApiField.FIELD: ApiField.ID,
                    ApiField.OPERATOR: "in",
                    ApiField.VALUE: eval_result.gt_dataset_ids,
                }
            ]
        if self.gt_dataset_infos is None:
            self.gt_dataset_infos = self.api.dataset.get_list(
                eval_result.gt_project_id,
                filters=filters,
                recursive=True,
            )
        eval_result.gt_dataset_infos = self.gt_dataset_infos
        filters = [
            {
                ApiField.FIELD: ApiField.NAME,
                ApiField.OPERATOR: "in",
                ApiField.VALUE: [ds.name for ds in self.gt_dataset_infos],
            }
        ]
        eval_result.pred_dataset_infos = self.api.dataset.get_list(
            eval_result.pred_project_id, filters=filters, recursive=True
        )

        # get train task info
        train_info = eval_result.train_info
        if train_info:
            train_task_id = train_info.get("app_session_id")
            if train_task_id:
                eval_result.task_info = self.api.task.get_info_by_id(int(train_task_id))

    def visualize(self):
        if self.renderer is None:
            layout = self._create_layout()
            self.renderer = Renderer(layout, self.workdir, report_name=self.report_name)
        return self.renderer.visualize()

    def upload_results(self, team_id: int, remote_dir: str, progress=None):
        if self.renderer is None:
            raise RuntimeError("Visualize first")
        return self.renderer.upload_results(self.api, team_id, remote_dir, progress)

    def _create_layout(self):
        raise NotImplementedError("Implement this method in a subclass")

    def _get_or_create_diff_project(self) -> Tuple[ProjectInfo, List, bool]:
        """
        Get or create a project for diff visualizations.
        Dataset hierarchy is copied from the prediction project.
        """

        pred_ds_id_to_diff_ds_info = {}
        diff_ds_infos = []

        def _get_or_create_diff_dataset(pred_dataset_id, pred_datasets):
            if pred_dataset_id in pred_ds_id_to_diff_ds_info:
                return pred_ds_id_to_diff_ds_info[pred_dataset_id]
            pred_dataset = pred_datasets[pred_dataset_id]
            if pred_dataset.parent_id is None:
                diff_dataset = self.api.dataset.create(project_info.id, pred_dataset.name)
            else:
                parent_dataset = _get_or_create_diff_dataset(pred_dataset.parent_id, pred_datasets)
                diff_dataset = self.api.dataset.create(
                    project_info.id,
                    pred_dataset.name,
                    parent_id=parent_dataset.id,
                )
            pred_ds_id_to_diff_ds_info[pred_dataset_id] = diff_dataset
            diff_ds_infos.append(diff_dataset)
            return diff_dataset

        is_existed = False
        project_name = self._generate_diff_project_name(self.eval_result.pred_project_info.name)
        workspace_id = self.eval_result.pred_project_info.workspace_id
        project_info = self.api.project.create(
            workspace_id, project_name, change_name_if_conflict=True
        )
        pred_datasets = {ds.id: ds for ds in self.eval_result.pred_dataset_infos}
        for dataset in pred_datasets:
            _get_or_create_diff_dataset(dataset, pred_datasets)
        return project_info, diff_ds_infos, is_existed

    def _generate_diff_project_name(self, pred_project_name):
        return "[diff]: " + pred_project_name

    def _create_explore_modal_table(
        self, columns_number=3, click_gallery_id=None, hover_text=None
    ) -> GalleryWidget:
        gallery = GalleryWidget(
            "all_predictions_modal_gallery",
            is_modal=True,
            columns_number=columns_number,
            click_gallery_id=click_gallery_id,
            opacity=self.ann_opacity,
        )
        gallery.set_project_meta(self.eval_results[0].filtered_project_meta)
        if hover_text:
            gallery.add_image_left_header(hover_text)
        return gallery

    def _create_diff_modal_table(self, columns_number=3) -> GalleryWidget:
        gallery = GalleryWidget(
            "diff_predictions_modal_gallery",
            is_modal=True,
            columns_number=columns_number,
            opacity=self.ann_opacity,
        )
        gallery.set_project_meta(self.eval_results[0].filtered_project_meta)
        return gallery

    def _get_filtered_project_meta(self, eval_result) -> ProjectMeta:
        remove_classes = []
        meta = eval_result.pred_project_meta.clone()
        if eval_result.classes_whitelist:
            for obj_class in meta.obj_classes:
                if obj_class.name not in eval_result.classes_whitelist:
                    remove_classes.append(obj_class.name)
            if remove_classes:
                meta = meta.delete_obj_classes(remove_classes)
        return meta

    def _update_match_data(
        self,
        gt_image_id: int,
        gt_image_info: ImageInfo = None,
        pred_image_info: ImageInfo = None,
        diff_image_info: ImageInfo = None,
        gt_annotation: Annotation = None,
        pred_annotation: Annotation = None,
        diff_annotation: Annotation = None,
    ):
        match_data = self.eval_result.matched_pair_data.get(gt_image_id, None)
        if match_data is None:
            self.eval_result.matched_pair_data[gt_image_id] = MatchedPairData(
                gt_image_info=gt_image_info,
                pred_image_info=pred_image_info,
                diff_image_info=diff_image_info,
                gt_annotation=gt_annotation,
                pred_annotation=pred_annotation,
                diff_annotation=diff_annotation,
            )
        else:
            for attr, value in {
                "gt_image_info": gt_image_info,
                "pred_image_info": pred_image_info,
                "diff_image_info": diff_image_info,
                "gt_annotation": gt_annotation,
                "pred_annotation": pred_annotation,
                "diff_annotation": diff_annotation,
            }.items():
                if value is not None:
                    setattr(match_data, attr, value)
