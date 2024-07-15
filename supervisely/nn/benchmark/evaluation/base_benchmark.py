import os
from typing import Union, List
from tqdm import tqdm

import supervisely as sly
from supervisely.nn.inference import SessionJSON
from supervisely.nn.benchmark.evaluation import BaseEvaluator


class BaseBenchmark:
    def __init__(
            self,
            api: sly.Api,
            gt_project_id: int,
            gt_dataset_ids: List[int] = None,
            output_dir: str = "./benchmark",
        ):
        self.api = api
        self.session: SessionJSON = None
        self.gt_project_info = api.project.get_info_by_id(gt_project_id)
        self.dt_project_info = None
        self.gt_dataset_ids = gt_dataset_ids
        self.output_dir = output_dir
        self.team_id = sly.env.team_id()
        self.evaluator: BaseEvaluator = None

    def run_evaluation(
            self,
            model_session: Union[int, str, SessionJSON],
            inference_settings = None,
            output_project_id = None,
            batch_size: int = 8,
            cache_project: bool = False
            ):
        self.session = self._init_model_session(model_session, inference_settings)
        self._eval_inference_info = self._run_inference(output_project_id, batch_size, cache_project)
        self.evaluate(self.dt_project_info.id)
    
    def run_inference(
            self,
            model_session: Union[int, str, SessionJSON],
            inference_settings = None,
            output_project_id = None,
            batch_size: int = 8,
            cache_project: bool = False
            ):
        self.session = self._init_model_session(model_session, inference_settings)
        self._eval_inference_info = self._run_inference(output_project_id, batch_size, cache_project)

    def _run_inference(
            self,
            output_project_id = None,
            batch_size: int = 8,
            cache_project: bool = False,
            ):
        model_info = self._fetch_model_info()

        if output_project_id is None:
            dt_project_name = self._generate_dt_project_name(self.gt_project_info.name, model_info)
            dt_wrokspace_id = self.gt_project_info.workspace_id
            dt_project_info = self.api.project.create(dt_wrokspace_id, dt_project_name, change_name_if_conflict=True)
            output_project_id = dt_project_info.id
        else:
            dt_project_info = self.api.project.get_info_by_id(output_project_id)

        self.dt_project_info = dt_project_info

        iterator = self.session.inference_project_id_async(
            self.gt_project_info.id,
            self.gt_dataset_ids,
            output_project_id=output_project_id,
            cache_project_on_model=cache_project,
            batch_size=batch_size,
        )

        for _ in tqdm(iterator):
            pass

        inference_info = {
            "gt_project_id": self.gt_project_info.id,
            "gt_dataset_ids": self.gt_dataset_ids,
            "dt_project_id": output_project_id,
            "batch_size": batch_size,
            **model_info,
        }        
        return inference_info
    
    def evaluate(self, dt_project_id):
        self.dt_project_info = self.api.project.get_info_by_id(dt_project_id)
        gt_project_path, dt_project_path = self._download_projects()
        self.evaluator = self._init_evaluator(gt_project_path, dt_project_path)
        self.evaluator.result_dir = self.get_eval_results_dir()
        eval_dir = self._evaluate()
        info_path = os.path.join(self.get_eval_results_dir(), "inference_info.json")
        sly.json.dump_json_file(self._eval_inference_info, info_path)
        return eval_dir
    
    def _evaluate(self):
        eval_dir = self.evaluator.evaluate()
        return eval_dir

    def get_eval_results_dir(self) -> str:
        return os.path.join(self.get_base_dir(), "eval_results")
    
    def upload_eval_results(self, remote_dir: str):
        eval_dir = self.get_eval_results_dir()
        assert not sly.fs.dir_empty(eval_dir), f"The result dir ({eval_dir}) is empty. You should run evaluation before uploading results."
        self.api.file.upload_directory(self.team_id, eval_dir, remote_dir)

    def run_speedtest(self):
        pass

    def upload_speedtest_results(self, remote_dir: str):
        pass
    
    # TODO: get_evaluator_class
    def _init_evaluator(self) -> BaseEvaluator:
        raise NotImplementedError()

    def get_base_dir(self):
        return os.path.join(self.output_dir, self.dt_project_info.name)

    def get_project_paths(self):
        base_dir = self.get_base_dir()
        gt_path = os.path.join(base_dir, "gt_project")
        dt_path = os.path.join(base_dir, "dt_project")
        return gt_path, dt_path
    
    def _generate_dt_project_name(self, gt_project_name, model_info):
        dt_project_name = gt_project_name + " - " + model_info["model_name"]
        if model_info.get("checkpoint_name"):
            # add checkpoint_id
            dt_project_name += f' ({model_info["checkpoint_name"]})'

    def _download_projects(self):
        gt_path, dt_path = self.get_project_paths()
        if not os.path.exists(gt_path):
            print(f"GT annotations will be downloaded to: {gt_path}")
            sly.download_project(
                self.api,
                self.gt_project_info.id,
                gt_path,
                dataset_ids=self.gt_dataset_ids,
                log_progress=True,
                save_images=False,
                save_image_info=True
                )
        else:
            print(f"GT annotations already exist: {gt_path}")
        if not os.path.exists(dt_path):
            print(f"DT annotations will be downloaded to: {dt_path}")
            sly.download_project(self.api,
            self.dt_project_info,
            dt_path,
            log_progress=True,
            save_images=False,
            save_image_info=True
            )
        else:
            print(f"DT annotations already exist: {dt_path}")
        _dump_project_info(self.gt_project_info, gt_path)
        _dump_project_info(self.dt_project_info, dt_path)
        return gt_path, dt_path
    
    def _fetch_model_info(self):
        deploy_info = self.session.get_deploy_info()
        task_info = self.api.task.get_info_by_id(self.session.task_id)
        app_info = task_info["meta"]["app"]
        app_info = {
            "name": app_info["name"],
            "version": app_info["version"],
            "id": app_info["id"],
        }
        model_info = {
            **deploy_info,
            "inference_settings": self.session.inference_settings,
            "app_info": app_info,
        }
        return model_info

    def _init_model_session(
            self,
            model_session: Union[int, str, SessionJSON],
            inference_settings: dict = None
            ):
        if isinstance(model_session, int):
            session = SessionJSON(self.api, model_session)
        elif isinstance(model_session, str):
            session = SessionJSON(self.api, session_url=model_session)
        elif isinstance(model_session, SessionJSON):
            session = model_session
        else:
            raise ValueError(f"Unsupported type of 'model_session' argument: {type(model_session)}")
        
        if inference_settings is not None:
            session.set_inference_settings(inference_settings)
        return session


def _dump_project_info(project_info: sly.ProjectInfo, project_path):
    project_info_path = os.path.join(project_path, "project_info.json")
    with open(project_info_path, 'w') as f:
        sly.json.dump_json_file(project_info._asdict(), f, indent=2)
    return project_info_path
