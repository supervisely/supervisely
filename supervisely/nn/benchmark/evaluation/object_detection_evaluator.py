import os

from supervisely.io.json import dump_json_file
from supervisely.nn.benchmark.coco_utils import read_coco_datasets, sly2coco
from supervisely.nn.benchmark.evaluation import BaseEvaluator
from supervisely.nn.benchmark.evaluation.coco import calculate_metrics


class ObjectDetectionEvaluator(BaseEvaluator):
    def evaluate(self):
        self.cocoGt_json, self.cocoDt_json = self._convert_to_coco()
        self.cocoGt, self.cocoDt = read_coco_datasets(self.cocoGt_json, self.cocoDt_json)
        with self.pbar(message="Evaluation: Calculating metrics", total=10) as p:
            self.eval_data = calculate_metrics(
                self.cocoGt, self.cocoDt, iouType="bbox", progress_cb=p.update
            )
        self._dump_eval_results()

    def _convert_to_coco(self):
        cocoGt_json = sly2coco(
            self.gt_project_path,
            is_dt_dataset=False,
            accepted_shapes=["rectangle"],
            progress=self.pbar,
            classes_whitelist=self.classes_whitelist,
        )
        cocoDt_json = sly2coco(
            self.dt_project_path,
            is_dt_dataset=True,
            accepted_shapes=["rectangle"],
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
        ), "Classes in GT and Pred projects must be the same"
        assert [x["id"] for x in cocoDt_json["images"]] == [x["id"] for x in cocoGt_json["images"]]
        return cocoGt_json, cocoDt_json

    def _dump_eval_results(self):
        cocoGt_path, cocoDt_path, eval_data_path = self._get_eval_paths()
        dump_json_file(self.cocoGt_json, cocoGt_path, indent=None)
        dump_json_file(self.cocoDt_json, cocoDt_path, indent=None)
        self._dump_pickle(self.eval_data, eval_data_path)

    def _get_eval_paths(self):
        base_dir = self.result_dir
        cocoGt_path = os.path.join(base_dir, "cocoGt.json")
        cocoDt_path = os.path.join(base_dir, "cocoDt.json")
        eval_data_path = os.path.join(base_dir, "eval_data.pkl")
        return cocoGt_path, cocoDt_path, eval_data_path
