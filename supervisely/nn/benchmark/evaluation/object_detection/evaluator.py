import os
from pycocotools.coco import COCO

import supervisely as sly
from supervisely.nn.benchmark.evaluation import BaseEvaluator
from supervisely.nn.benchmark.evaluation.object_detection import calculate_metrics
from supervisely.nn.benchmark.sly2coco import sly2coco


class ObjectDetectionEvaluator(BaseEvaluator):
    def evaluate(self) -> str:
        self.cocoGt_json, self.cocoDt_json = self.convert_to_coco()
        cocoGt, cocoDt = read_coco_datasets(self.cocoGt_json, self.cocoDt_json)
        self.eval_data = calculate_metrics(cocoGt, cocoDt)
        self.dump_eval_results()

    def convert_to_coco(self):
        cocoGt_json = sly2coco(self.gt_project_path, is_dt_dataset=False, accepted_shapes=["rectangle"])
        cocoDt_json = sly2coco(self.dt_project_path, is_dt_dataset=True, accepted_shapes=["rectangle"])
        assert cocoDt_json['categories'] == cocoGt_json['categories']
        assert [x['id'] for x in cocoDt_json['images']] == [x['id'] for x in cocoGt_json['images']]
        return cocoGt_json, cocoDt_json

    def get_eval_paths(self):
        base_dir = self.result_dir
        cocoGt_path = os.path.join(base_dir, "cocoGt.json")
        cocoDt_path = os.path.join(base_dir, "cocoDt.json")
        eval_data_path = os.path.join(base_dir, "eval_data.pkl")
        eval_info_path = os.path.join(base_dir, "info.json")
        return cocoGt_path, cocoDt_path, eval_data_path

    def dump_eval_results(self):
        cocoGt_path, cocoDt_path, eval_data_path, eval_info_path = self.get_eval_paths()
        sly.json.dump_json_file(self.cocoGt_json, cocoGt_path, indent=None)
        sly.json.dump_json_file(self.cocoDt_json, cocoDt_path, indent=None)
        # sly.json.dump_json_file(eval_info, eval_info_path, indent=2)
        self._dump_pickle(self.eval_data, eval_data_path)


def read_coco_datasets(cocoGt_json, cocoDt_json):
    if isinstance(cocoGt_json, str):
        cocoGt_json = sly.json.load_json_file(cocoGt_json)
    if isinstance(cocoDt_json, str):
        cocoDt_json = sly.json.load_json_file(cocoDt_json)
    cocoGt = COCO()
    cocoGt.dataset = cocoGt_json
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes(cocoDt_json['annotations'])
    return cocoGt, cocoDt

