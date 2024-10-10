import os

from supervisely.io.json import dump_json_file
from supervisely.nn.benchmark.coco_utils import read_coco_datasets, sly2coco
from supervisely.nn.benchmark.evaluation import BaseEvaluator
from supervisely.nn.benchmark.evaluation.coco import calculate_metrics
from pathlib import Path


class InstanceSegmentationEvaluator(BaseEvaluator):
    EVALUATION_PARAMS_YAML_PATH = f"{Path(__file__).parent}/coco/evaluation_params.yaml"

    def evaluate(self):
        try:
            self.cocoGt_json, self.cocoDt_json = self._convert_to_coco()
        except AssertionError as e:
            raise ValueError(
                f"{e}. Please make sure that your GT and DT projects are correct. "
                "If GT project has nested datasets and DT project was crated with NN app, "
                "try to use newer version of NN app."
            )

        self._dump_datasets()
        self.cocoGt, self.cocoDt = read_coco_datasets(self.cocoGt_json, self.cocoDt_json)
        with self.pbar(message="Evaluation: Calculating metrics", total=5) as p:
            self.eval_data = calculate_metrics(
                self.cocoGt,
                self.cocoDt,
                iouType="segm",
                progress_cb=p.update,
                evaluation_params=self.evaluation_params,
            )
        self._dump_eval_results()

    @classmethod
    def validate_evaluation_params(cls, evaluation_params: dict) -> None:
        iou_threshold = evaluation_params.get("iou_threshold")
        if iou_threshold is not None:
            assert iou_threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], (
                f"iou_threshold must be one of [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "
                f"but got {iou_threshold}"
            )

    def _convert_to_coco(self):
        cocoGt_json = sly2coco(
            self.gt_project_path,
            is_dt_dataset=False,
            accepted_shapes=["polygon", "bitmap"],
            progress=self.pbar,
            classes_whitelist=self.classes_whitelist,
        )
        cocoDt_json = sly2coco(
            self.dt_project_path,
            is_dt_dataset=True,
            accepted_shapes=["polygon", "bitmap"],
            progress=self.pbar,
            classes_whitelist=self.classes_whitelist,
        )
        if len(cocoGt_json["annotations"]) == 0:
            raise ValueError("Not found any annotations in GT project")
        if len(cocoDt_json["annotations"]) == 0:
            raise ValueError(
                "Not found any predictions. "
                "Please make sure that your model produces predictions."
            )
        assert (
            cocoDt_json["categories"] == cocoGt_json["categories"]
        ), "Object classes in GT and DT projects are different"
        assert [f'{x["dataset"]}/{x["file_name"]}' for x in cocoDt_json["images"]] == [
            f'{x["dataset"]}/{x["file_name"]}' for x in cocoGt_json["images"]
        ], "Images in GT and DT projects are different"
        return cocoGt_json, cocoDt_json

    def _dump_datasets(self):
        cocoGt_path, cocoDt_path, eval_data_path = self._get_eval_paths()
        dump_json_file(self.cocoGt_json, cocoGt_path, indent=None)
        dump_json_file(self.cocoDt_json, cocoDt_path, indent=None)

    def _dump_eval_results(self):
        cocoGt_path, cocoDt_path, eval_data_path = self._get_eval_paths()
        self._dump_pickle(self.eval_data, eval_data_path)

    def _get_eval_paths(self):
        base_dir = self.result_dir
        cocoGt_path = os.path.join(base_dir, "cocoGt.json")
        cocoDt_path = os.path.join(base_dir, "cocoDt.json")
        eval_data_path = os.path.join(base_dir, "eval_data.pkl")
        return cocoGt_path, cocoDt_path, eval_data_path
