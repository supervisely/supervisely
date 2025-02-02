import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from supervisely.annotation.annotation import ProjectMeta
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import SlyTqdm
from supervisely.io.env import team_id
from supervisely.io.fs import dir_empty, mkdir
from supervisely.io.json import load_json_file
from supervisely.nn.benchmark.object_detection.metric_provider import MetricProvider
from supervisely.nn.benchmark.visualization.vis_click_data import ClickData, IdMapper
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly

# class ImageComparisonData:
#     def __init__(
#         self,
#         gt_image_info: ImageInfo = None,
#         pred_image_info: ImageInfo = None,
#         diff_image_info: ImageInfo = None,
#         gt_annotation: Annotation = None,
#         pred_annotation: Annotation = None,
#         diff_annotation: Annotation = None,
#     ):
#         self.gt_image_info = gt_image_info
#         self.pred_image_info = pred_image_info
#         self.diff_image_info = diff_image_info
#         self.gt_annotation = gt_annotation
#         self.pred_annotation = pred_annotation
#         self.diff_annotation = diff_annotation


class EvalResult:

    def __init__(
        self,
        eval_dir: str,
        workdir: str,
        api: Api,
        progress: Optional[SlyTqdm] = None,
    ):
        from pycocotools.coco import COCO  # pylint: disable=import-error

        self.eval_dir = eval_dir
        self.report_path = Path(eval_dir, "visualizations", "template.vue").as_posix()
        self.workdir = workdir
        self.api = api
        self.team_id = team_id()
        self.local_dir = str(Path(self.workdir, self.eval_dir.lstrip("/")))
        self.progress = progress or tqdm_sly

        self.coco_gt: COCO = None
        self.coco_dt: COCO = None
        self.inference_info: Dict = None
        self.speedtest_info: Dict = None
        self.eval_data: Dict = None
        self.mp: MetricProvider = None
        self.df_score_profile: pd.DataFrame = None
        self.dfsp_down: pd.DataFrame = None
        self.f1_optimal_conf: float = None
        self.click_data: ClickData = None
        # self.comparison_data: Dict[int, ImageComparisonData] = {}
        self.color = None

        self._gt_project_info = None
        self._gt_project_meta = None
        self._gt_dataset_infos = None
        self._dt_project_id = None
        self._dt_project_meta = None

        self._load_eval_data()
        self._read_eval_data()

    @property
    def cv_task(self):
        return self.inference_info.get("task_type")

    @property
    def name(self) -> str:
        model_name = self.inference_info.get("model_name", self.eval_dir)
        return self.inference_info.get("deploy_params", {}).get("checkpoint_name", model_name)

    @property
    def model_name(self) -> str:
        if not self.name:
            return
        if len(self.name) > 20:
            return self.name[:9] + "..." + self.name[-6:]
        return self.name

    @property
    def gt_project_id(self) -> int:
        return self.inference_info.get("gt_project_id")

    @property
    def gt_project_info(self) -> ProjectInfo:
        if self._gt_project_info is None:
            gt_project_id = self.inference_info.get("gt_project_id")
            self._gt_project_info = self.api.project.get_info_by_id(gt_project_id)
        return self._gt_project_info

    @property
    def gt_project_meta(self) -> ProjectMeta:
        if self._gt_project_meta is None:
            self._gt_project_meta = ProjectMeta.from_json(
                self.api.project.get_meta(self.gt_project_id)
            )
        return self._gt_project_meta

    @property
    def gt_dataset_ids(self) -> List[int]:
        return self.inference_info.get("gt_dataset_ids", None)

    @property
    def gt_dataset_infos(self) -> List[DatasetInfo]:
        if self._gt_dataset_infos is None:
            filters = None
            if self.gt_dataset_ids is not None:
                filters = [
                    {
                        ApiField.FIELD: ApiField.ID,
                        ApiField.OPERATOR: "in",
                        ApiField.VALUE: self.gt_dataset_ids,
                    }
                ]
            self._gt_dataset_infos = self.api.dataset.get_list(
                self.gt_project_id,
                filters=filters,
                recursive=True,
            )
        return self._gt_dataset_infos

    @property
    def dt_project_id(self):
        if self._dt_project_id is None:
            self._dt_project_id = self.inference_info.get("dt_project_id")
        return self._dt_project_id

    @property
    def dt_project_meta(self):
        if self._dt_project_meta is None:
            self._dt_project_meta = ProjectMeta.from_json(
                self.api.project.get_meta(self.dt_project_id)
            )
        return self._dt_project_meta

    @property
    def train_info(self):
        return self.inference_info.get("train_info", None)  # TODO: check

    @property
    def gt_images_ids(self):
        return self.inference_info.get("gt_images_ids", None)  # TODO: check

    @property
    def classes_whitelist(self):
        return self.inference_info.get("inference_settings", {}).get("classes", [])  # TODO: check

    def _load_eval_data(self):
        if not dir_empty(self.local_dir):
            return
        if not self.api.storage.dir_exists(self.team_id, self.eval_dir):
            raise ValueError(f"Directory {self.eval_dir} not found in storage.")
        mkdir(self.local_dir)
        dir_name = Path(self.eval_dir).name
        with self.progress(
            message=f"Downloading evaluation data from {dir_name}",
            total=self.api.storage.get_directory_size(self.team_id, self.eval_dir),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            self.api.storage.download_directory(
                self.team_id, self.eval_dir, self.local_dir, progress_cb=pbar.update
            )

    # def _load_projects(self):
    #     projects_dir = Path(self.local_dir, "projects")
    #     items_total = self.gt_images_ids
    #     if items_total is None:
    #         items_total = sum(self.gt_dataset_infos, key=lambda x: x.items_count)
    #     with self.progress(
    #         message=f"Downloading GT project {self.gt_project_info.name} and datasets",
    #         total=items_total,
    #     ) as pbar:
    #         download_project(
    #             self.api,
    #             self.gt_project_info.id,
    #             str(projects_dir),
    #             dataset_ids=self.gt_dataset_ids,
    #             progress_cb=pbar.update,
    #         )

    def _read_eval_data(self):
        from pycocotools.coco import COCO  # pylint: disable=import-error

        gt_path = str(Path(self.local_dir, "evaluation", "cocoGt.json"))
        dt_path = str(Path(self.local_dir, "evaluation", "cocoDt.json"))
        coco_gt, coco_dt = COCO(gt_path), COCO(dt_path)
        self.coco_gt = coco_gt
        self.coco_dt = coco_dt
        self.eval_data = pickle.load(
            open(Path(self.local_dir, "evaluation", "eval_data.pkl"), "rb")
        )
        self.inference_info = load_json_file(
            Path(self.local_dir, "evaluation", "inference_info.json")
        )
        speedtest_info_path = Path(self.local_dir, "speedtest", "speedtest.json")
        if speedtest_info_path.exists():
            self.speedtest_info = load_json_file(
                Path(self.local_dir, "speedtest", "speedtest.json")
            )

        self.mp = MetricProvider(
            self.eval_data,
            self.coco_gt,
            self.coco_dt,
        )
        self.mp.calculate()

        self.df_score_profile = pd.DataFrame(
            self.mp.confidence_score_profile(), columns=["scores", "precision", "recall", "f1"]
        )

        # downsample
        if len(self.df_score_profile) > 5000:
            self.dfsp_down = self.df_score_profile.iloc[:: len(self.df_score_profile) // 1000]
        else:
            self.dfsp_down = self.df_score_profile

        self.f1_optimal_conf = self.mp.get_f1_optimal_conf()[0]
        if self.f1_optimal_conf is None:
            self.f1_optimal_conf = 0.01
            logger.warning("F1 optimal confidence cannot be calculated. Using 0.01 as default.")

        # Click data
        gt_id_mapper = IdMapper(self.coco_gt.dataset)
        dt_id_mapper = IdMapper(self.coco_dt.dataset)

        self.click_data = ClickData(self.mp.m, gt_id_mapper, dt_id_mapper)

    # def _update_comparison_data(
    #     self,
    #     gt_image_id: int,
    #     gt_image_info: ImageInfo = None,
    #     pred_image_info: ImageInfo = None,
    #     diff_image_info: ImageInfo = None,
    #     gt_annotation: Annotation = None,
    #     pred_annotation: Annotation = None,
    #     diff_annotation: Annotation = None,
    # ):
    #     comparison_data = self.comparison_data.get(gt_image_id, None)
    #     if comparison_data is None:
    #         self.comparison_data[gt_image_id] = ImageComparisonData(
    #             gt_image_info=gt_image_info,
    #             pred_image_info=pred_image_info,
    #             diff_image_info=diff_image_info,
    #             gt_annotation=gt_annotation,
    #             pred_annotation=pred_annotation,
    #             diff_annotation=diff_annotation,
    #         )
    #     else:
    #         for attr, value in {
    #             "gt_image_info": gt_image_info,
    #             "pred_image_info": pred_image_info,
    #             "diff_image_info": diff_image_info,
    #             "gt_annotation": gt_annotation,
    #             "pred_annotation": pred_annotation,
    #             "diff_annotation": diff_annotation,
    #         }.items():
    #             if value is not None:
    #                 setattr(comparison_data, attr, value)
