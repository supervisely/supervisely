import random
from typing import Dict, List, Optional, Tuple

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.nn.benchmark.base_evaluator import BaseEvalResult
from supervisely.nn.benchmark.visualization.renderer import Renderer
from supervisely.nn.benchmark.visualization.widgets import GalleryWidget
from supervisely.project.project_meta import ProjectMeta
from supervisely.nn.benchmark.cv_tasks import CVTask


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

    def get_click_data(self) -> Optional[Dict]:
        if not self.clickable:
            return

        res = {}

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}
        for key, v in self.eval_result.click_data.objects_by_class.items():
            res["clickData"][key] = {}
            res["clickData"][key]["imagesIds"] = []

            # tmp = defaultdict(list)
            img_ids = set()
            obj_ids = set()

            res["clickData"][key][
                "title"
            ] = f"{key} class: {len(v)} object{'s' if len(v) > 1 else ''}"

            for x in v:
                img_ids.add(x["dt_img_id"])
                obj_ids.add(x["dt_obj_id"])

            res["clickData"][key]["imagesIds"] = list(img_ids)
            res["clickData"][key]["filters"] = [
                {
                    "type": "tag",
                    "tagId": "confidence",
                    "value": [self.eval_result.mp.f1_optimal_conf, 1],
                },
                {"type": "tag", "tagId": "outcome", "value": "TP"},
                {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
            ]

        return res


class BaseVisualizer:

    def __init__(
        self,
        api: Api,
        eval_results: List[BaseEvalResult],
        workdir="./visualizations",
    ):
        self.api = api
        self.workdir = workdir
        self.eval_result = eval_results[0]  # for evaluation
        self.eval_results = eval_results  # for comparison

        self.renderer = None
        self.gt_project_info = None
        self.gt_project_meta = None
        self.gt_dataset_infos = None

        self.cv_task = None

        for eval_result in self.eval_results:
            self._get_eval_project_infos(eval_result)

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

        # set filtered project meta
        eval_result.filtered_project_meta = self._get_filtered_project_meta(
            eval_result.pred_project_meta, eval_result
        )

        # get dataset infos
        filters = None
        if eval_result.gt_dataset_ids is not None:
            filters = [{"field": "id", "operator": "in", "value": eval_result.gt_dataset_ids}]
        if self.gt_dataset_infos is None:
            self.gt_dataset_infos = self.api.dataset.get_list(
                eval_result.gt_project_id,
                filters=filters,
                recursive=True,
            )
        eval_result.gt_dataset_infos = self.gt_dataset_infos
        eval_result.pred_dataset_infos = self.api.dataset.get_list(
            eval_result.pred_project_id, recursive=True
        )

        # get train task info
        train_info = eval_result.train_info
        if train_info:
            train_task_id = train_info.get("app_session_id")
            if train_task_id:
                eval_result.task_info = self.api.task.get_info_by_id(int(train_task_id))

        # get sample images with annotations for visualization
        pred_dataset = random.choice(eval_result.pred_dataset_infos)
        eval_result.sample_images = self.api.image.get_list(dataset_id=pred_dataset.id, limit=9)
        image_ids = [x.id for x in eval_result.sample_images]
        eval_result.sample_anns = self.api.annotation.download_batch(pred_dataset.id, image_ids)

    def visualize(self):
        if self.renderer is None:
            layout = self._create_layout()
            self.renderer = Renderer(layout, self.workdir)
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

        project_name = self._generate_diff_project_name(self.eval_result.pred_project_info.name)
        workspace_id = self.eval_result.pred_project_info.workspace_id
        project_info = self.api.project.get_info_by_name(
            workspace_id, project_name, raise_error=False
        )
        is_existed = project_info is not None
        if not is_existed:
            project_info = self.api.project.create(
                workspace_id, project_name, change_name_if_conflict=True
            )
            pred_datasets = {ds.id: ds for ds in self.eval_result.pred_dataset_infos}
            for dataset in pred_datasets:
                _get_or_create_diff_dataset(dataset, pred_datasets)
        return project_info, diff_ds_infos, is_existed

    def _generate_diff_project_name(self, pred_project_name):
        return "[diff]: " + pred_project_name

    def _create_explore_modal_table(self, columns_number=3, click_gallery_id=None) -> GalleryWidget:
        gallery = GalleryWidget(
            "all_predictions_modal_gallery",
            is_modal=True,
            columns_number=columns_number,
            click_gallery_id=click_gallery_id,
            opacity=self.ann_opacity,
        )
        gallery.set_project_meta(self.eval_results[0].filtered_project_meta)
        gallery.add_image_left_header("Compare with GT")
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

    def _get_filtered_project_meta(self, meta: ProjectMeta, eval_result) -> ProjectMeta:
        remove_classes = []
        if eval_result.classes_whitelist:
            for obj_class in meta.obj_classes:
                if obj_class.name not in eval_result.classes_whitelist:
                    remove_classes.append(obj_class.name)
            if remove_classes:
                meta = meta.delete_obj_classes(remove_classes)
        return meta
