from supervisely.nn.benchmark.comparison.base_comparison import BaseComparison
from supervisely.nn.benchmark.comparison.visualization.visualizer import (
    ComparisonVisualizer,
)
from supervisely.nn.benchmark.evaluation.coco.metric_provider import MetricProvider


class ObjectDetectionComparison(BaseComparison):
    def run_compare(self):
        pass

    def _initialize_loaders(self):
        return
        # for eval_data, coco_gt, coco_dt,
        # self.df_score_profile = pd.DataFrame(
        #     self.mp.confidence_score_profile(), columns=["scores", "precision", "recall", "f1"]
        # )

        # # downsample
        # if len(self.df_score_profile) > 5000:
        #     self.dfsp_down = self.df_score_profile.iloc[:: len(self.df_score_profile) // 1000]
        # else:
        #     self.dfsp_down = self.df_score_profile

        # self.f1_optimal_conf = self.mp.get_f1_optimal_conf()[0]
        # if self.f1_optimal_conf is None:
        #     self.f1_optimal_conf = 0.01
        #     logger.warn("F1 optimal confidence cannot be calculated. Using 0.01 as default.")

        # # Click data
        # gt_id_mapper = IdMapper(cocoGt_dataset)
        # dt_id_mapper = IdMapper(cocoDt_dataset)

        # self.click_data = ClickData(self.mp.m, gt_id_mapper, dt_id_mapper)
        # self.base_metrics = self.mp.base_metrics

        # self._objects_bindings = []


# TODO: check if model evaluations of the same project or datasets (if datasets are provided)
# ? TODO: handle the case when the models evaluated on different classes
# ? TODO: handle the case when the models evaluated on different datasets


# usage example:
# eval_dir_1 = "/model_benchmark/project_1/54601_serve_yolov8/evaluation/"
# eval_dir_2 = "/model_benchmark/project_1/54602_serve_yolov8/evaluation/"
# res_remote_dir = "/model_benchmark/project_1/comparison_results/"

# from supervisely.api.api import Api
# from supervisely.io import env

# team_id = env.team_id()
# api = Api()

# bm_comparison = ObjectDetectionComparison(api, [eval_dir_1, eval_dir_2])  # ? , local=True)
# bm_comparison.run_compare()
# bm_comparison.visualize()
# bm_comparison.upload_results(team_id, res_remote_dir)
# bm_comparison.get_report_link(team_id, res_remote_dir)
