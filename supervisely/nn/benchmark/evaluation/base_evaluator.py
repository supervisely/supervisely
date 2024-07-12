from typing import Union, List
from tqdm import tqdm

import supervisely as sly
from supervisely.nn.inference import SessionJSON


class BaseEvaluator:
    def __init__(
            self,
            model_session: SessionJSON,
            gt_project_id: int,
            gt_dataset_ids: List[int] = None,
            output_dir: str = "./eval_output",
        ):
        self.session = model_session
        self.gt_project_id = gt_project_id
        self.gt_dataset_ids = gt_dataset_ids
        self.output_dir = output_dir
        self.api = model_session.api
        self.dt_project_id = None
        self._batch_size = None
        self.eval_info = None

    def run_evaluation(
            self,
            dt_project_id = None,
            batch_size: int = 8,
            cache_project: bool = False
            ):
        # inference
        # - gt_project_id, dt_project_id, inference_settings, gt_dataset_ids, batch_size, cache_project
        # download_projects
        # - gt_project_id, dt_project_id
        # calculate_metrics
        # - GT anns (project), DT anns (project)
        # upload_results
        self.run_inference(dt_project_id, batch_size, cache_project)
        pass
    
    def run_inference(
            self,
            dt_project_id = None,
            batch_size: int = 8,
            cache_project: bool = False
            ):
        self._batch_size = batch_size
        self.eval_info = self._collect_evaluation_info()
        gt_project_info = self.api.project.get_info_by_id(self.gt_project_id)
        if dt_project_id is None:
            dt_project_name = gt_project_info.name + " - " + self.eval_info["model_name"]  #  + checkpoint_id
            if self.eval_info.get("checkpoint_name"):
                dt_project_name += f' ({self.eval_info["checkpoint_name"]})'
            dt_wrokspace_id = gt_project_info.workspace_id
            dt_project_info = self.api.project.create(dt_wrokspace_id, dt_project_name, change_name_if_conflict=True)
            dt_project_id = dt_project_info.id

        self.dt_project_id = dt_project_id

        iterator = self.session.inference_project_id_async(
            self.gt_project_id,
            self.gt_dataset_ids,
            output_project_id=dt_project_id,
            cache_project_on_model=cache_project,
            batch_size=batch_size,
        )

        for _ in tqdm(iterator):
            pass

        self.eval_info["dt_project_id"] = dt_project_id
        self.api.project.update_custom_data(dt_project_id, self.eval_info)
    
    def evaluate(self):
        raise NotImplementedError()
    
    def upload_results(self):
        raise NotImplementedError()
    
    def _collect_evaluation_info(self):
        deploy_info = self.session.get_deploy_info()
        task_info = self.api.task.get_info_by_id(self.session.task_id)
        app_info = task_info["meta"]["app"]
        app_info = {
            "name": app_info["name"],
            "version": app_info["version"],
            "id": app_info["id"],
        }
        eval_info = {
            "gt_project_id": self.gt_project_id,
            "gt_dataset_ids": self.gt_dataset_ids,
            **deploy_info,
            "inference_settings": self.session.inference_settings,
            "batch_size": self._batch_size,
            "app_info": app_info,
        }
        return eval_info
