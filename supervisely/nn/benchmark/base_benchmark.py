import os
from typing import List, Optional, Tuple, Union

import numpy as np

import supervisely as sly
from supervisely.api.project_api import ProjectInfo
from supervisely.io.fs import get_directory_size
from supervisely.nn.benchmark.evaluation import BaseEvaluator
from supervisely.nn.benchmark.visualization.visualizer import Visualizer
from supervisely.nn.inference import SessionJSON
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly


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
        self.diff_project_info = None
        self.gt_dataset_ids = gt_dataset_ids
        self.output_dir = output_dir
        self.team_id = sly.env.team_id()
        self.evaluator: BaseEvaluator = None
        self._eval_inference_info = None
        self._speedtest = None

    def _get_evaluator_class(self) -> type:
        raise NotImplementedError()

    @property
    def cv_task(self) -> str:
        raise NotImplementedError()

    def run_evaluation(
        self,
        model_session: Union[int, str, SessionJSON],
        inference_settings=None,
        output_project_id=None,
        batch_size: int = 8,
        cache_project_on_agent: bool = False,
    ):
        self.session = self._init_model_session(model_session, inference_settings)
        self._eval_inference_info = self._run_inference(output_project_id, batch_size, cache_project_on_agent)
        self.evaluate(self.dt_project_info.id)
        self._dump_eval_inference_info(self._eval_inference_info)

    def run_inference(
        self,
        model_session: Union[int, str, SessionJSON],
        inference_settings=None,
        output_project_id=None,
        batch_size: int = 8,
        cache_project_on_agent: bool = False,
    ):
        self.session = self._init_model_session(model_session, inference_settings)
        self._eval_inference_info = self._run_inference(output_project_id, batch_size, cache_project_on_agent)

    def _run_inference(
        self,
        output_project_id=None,
        batch_size: int = 8,
        cache_project_on_agent: bool = False,
    ):
        model_info = self._fetch_model_info(self.session)
        self.dt_project_info = self._get_or_create_dt_project(output_project_id, model_info)
        iterator = self.session.inference_project_id_async(
            self.gt_project_info.id,
            self.gt_dataset_ids,
            output_project_id=self.dt_project_info.id,
            cache_project_on_model=cache_project_on_agent,
            batch_size=batch_size,
        )
        for _ in tqdm_sly(iterator):
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
        self._evaluate(gt_project_path, dt_project_path)

    def _evaluate(self, gt_project_path, dt_project_path):
        eval_results_dir = self.get_eval_results_dir()
        self.evaluator = self._get_evaluator_class()(
            gt_project_path=gt_project_path,
            dt_project_path=dt_project_path,
            result_dir=eval_results_dir,
        )
        self.evaluator.evaluate()

    def run_speedtest(
        self,
        model_session: Union[int, str, SessionJSON],
        project_id: int,
        batch_sizes: list = (1, 8, 16),
        inference_settings: dict = None,
        num_iterations: int = 100,
        num_warmup: int = 3,
        cache_project_on_agent=False,
    ):
        self.session = self._init_model_session(model_session, inference_settings)
        self._speedtest = self._run_speedtest(
            project_id,
            batch_sizes=batch_sizes,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
            cache_project_on_agent=cache_project_on_agent,
        )
        self._dump_speedtest(self._speedtest)

    def _run_speedtest(
        self,
        project_id: int,
        batch_sizes: list = (1, 8, 16),
        num_iterations: int = 100,
        num_warmup: int = 3,
        cache_project_on_agent=False,
    ):
        model_info = self._fetch_model_info(self.session)
        speedtest_info = {
            "device": model_info["device"],
            "runtime": model_info["runtime"],
            "hardware": model_info["hardware"],
            "num_iterations": num_iterations,
        }
        benchmarks = []
        for bs in batch_sizes:
            sly.logger.debug(f"Running speedtest for batch_size={bs}")
            speedtest_results = []
            iterator = self.session.run_benchmark(
                project_id,
                batch_size=bs,
                num_iterations=num_iterations,
                num_warmup=num_warmup,
                cache_project_on_model=cache_project_on_agent,
            )
            for speedtest in tqdm_sly(iterator):
                speedtest_results.append(speedtest)
            assert len(speedtest_results) == num_iterations, "Speedtest failed to run all iterations."
            avg_speedtest, std_speedtest = self._calculate_speedtest_statistics(speedtest_results)
            benchmark = {
                "benchmark": avg_speedtest,
                "benchmark_std": std_speedtest,
                "batch_size": bs,
                **speedtest_info,
            }
            benchmarks.append(benchmark)
        speedtest = {
            "model_info": model_info,
            "speedtest": benchmarks,
        }
        return speedtest

    def get_base_dir(self):
        return os.path.join(self.output_dir, self.dt_project_info.name)

    def get_project_paths(self):
        base_dir = self.get_base_dir()
        gt_path = os.path.join(base_dir, "gt_project")
        dt_path = os.path.join(base_dir, "dt_project")
        return gt_path, dt_path

    def get_eval_results_dir(self) -> str:
        dir = os.path.join(self.get_base_dir(), "evaluation")
        os.makedirs(dir, exist_ok=True)
        return dir

    def get_speedtest_results_dir(self) -> str:
        checkpoint_name = self._speedtest["model_info"]["model_name"]
        dir = os.path.join(
            self.output_dir, "speedtest", checkpoint_name
        )  # TODO: use checkpoint_name instead of model_name
        os.makedirs(dir, exist_ok=True)
        return dir

    def upload_eval_results(self, remote_dir: str):
        eval_dir = self.get_eval_results_dir()
        assert not sly.fs.dir_empty(
            eval_dir
        ), f"The result dir {eval_dir!r} is empty. You should run evaluation before uploading results."
        self.api.file.upload_directory(
            self.team_id,
            eval_dir,
            remote_dir,
            replace_if_conflict=True,
            change_name_if_conflict=False,
        )

    def get_layout_results_dir(self) -> str:
        dir = os.path.join(self.get_base_dir(), "layout")
        os.makedirs(dir, exist_ok=True)
        return dir

    def upload_speedtest_results(self, remote_dir: str):
        speedtest_dir = self.get_speedtest_results_dir()
        assert not sly.fs.dir_empty(
            speedtest_dir
        ), f"Speedtest dir {speedtest_dir!r} is empty. You should run speedtest before uploading results."
        self.api.file.upload_directory(self.team_id, speedtest_dir, remote_dir)

    def _generate_dt_project_name(self, gt_project_name, model_info):
        dt_project_name = gt_project_name + " - " + model_info["checkpoint_name"]
        return dt_project_name

    def _generate_diff_project_name(self, dt_project_name):
        return "[diff]: " + dt_project_name

    def _get_or_create_dt_project(self, output_project_id, model_info) -> sly.ProjectInfo:
        if output_project_id is None:
            dt_project_name = self._generate_dt_project_name(self.gt_project_info.name, model_info)
            dt_wrokspace_id = self.gt_project_info.workspace_id
            dt_project_info = self.api.project.create(dt_wrokspace_id, dt_project_name, change_name_if_conflict=True)
            output_project_id = dt_project_info.id
        else:
            dt_project_info = self.api.project.get_info_by_id(output_project_id)
        return dt_project_info

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
                save_image_info=True,
            )
        else:
            print(f"GT annotations already exist: {gt_path}")
        if not os.path.exists(dt_path):
            print(f"DT annotations will be downloaded to: {dt_path}")
            sly.download_project(
                self.api,
                self.dt_project_info.id,
                dt_path,
                log_progress=True,
                save_images=False,
                save_image_info=True,
            )
        else:
            print(f"DT annotations already exist: {dt_path}")
        self._dump_project_info(self.gt_project_info, gt_path)
        self._dump_project_info(self.dt_project_info, dt_path)
        return gt_path, dt_path

    def _fetch_model_info(self, session: SessionJSON):
        deploy_info = session.get_deploy_info()
        if session.task_id is not None:
            task_info = self.api.task.get_info_by_id(session.task_id)
            app_info = task_info["meta"]["app"]
            app_info = {
                "name": app_info["name"],
                "version": app_info["version"],
                "id": app_info["id"],
            }
        else:
            sly.logger.warn("session.task_id is not set. App info will not be fetched.")
            app_info = None
        model_info = {
            **deploy_info,
            "inference_settings": session.inference_settings,
            "app_info": app_info,
        }
        return model_info

    def _init_model_session(self, model_session: Union[int, str, SessionJSON], inference_settings: dict = None):
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

    def _dump_project_info(self, project_info: sly.ProjectInfo, project_path):
        project_info_path = os.path.join(project_path, "project_info.json")
        sly.json.dump_json_file(project_info._asdict(), project_info_path, indent=2)
        return project_info_path

    def _dump_eval_inference_info(self, eval_inference_info):
        info_path = os.path.join(self.get_eval_results_dir(), "inference_info.json")
        sly.json.dump_json_file(eval_inference_info, info_path)
        return info_path

    def _dump_speedtest(self, speedtest):
        path = os.path.join(self.get_speedtest_results_dir(), "speedtest.json")
        sly.json.dump_json_file(speedtest, path, indent=2)
        return path

    def _calculate_speedtest_statistics(self, speedtest_results: list):
        x = [[s[k] for s in speedtest_results] for k in speedtest_results[0].keys()]
        x = np.array(x, dtype=float)
        avg = x.mean(1)
        std = x.std(1)
        avg_speedtest = {
            k: float(avg[i]) if not np.isnan(avg[i]).any() else None for i, k in enumerate(speedtest_results[0].keys())
        }
        std_speedtest = {
            k: float(std[i]) if not np.isnan(std[i]).any() else None for i, k in enumerate(speedtest_results[0].keys())
        }
        return avg_speedtest, std_speedtest

    def _format_speedtest_as_table(self, speedtest: dict):
        benchmarks = speedtest["speedtest"]
        rows = []
        for benchmark in benchmarks:
            row = benchmark.copy()
            avg = row.pop("benchmark")
            std = row.pop("benchmark_std")
            for key in avg.keys():
                row[key + "_avg"] = avg[key]
                row[key + "_std"] = std[key]
            rows.append(row)
        return rows

    def visualize(self, dt_project_id=None):
        if dt_project_id is None:
            if self.dt_project_info is None:
                raise RuntimeError(
                    "The prediction dt_project was not initialized. Please run evaluation or find the ready project."
                )
        else:
            self.dt_project_info = self.api.project.get_info_by_id(dt_project_id)

        eval_dir = self.get_eval_results_dir()
        assert not sly.fs.dir_empty(
            eval_dir
        ), f"The result dir {eval_dir!r} is empty. You should run evaluation before uploading results."

        self.diff_project_info, was_before = self._get_or_create_diff_project()
        vis = Visualizer(self)
        if not was_before:
            vis.update_diff_annotations()
        vis.visualize()

    def _get_or_create_diff_project(self) -> Tuple[sly.ProjectInfo, bool]:
        diff_project_name = self._generate_diff_project_name(self.dt_project_info.name)
        diff_workspace_id = self.dt_project_info.workspace_id
        diff_project_info = self.api.project.get_info_by_name(diff_workspace_id, diff_project_name, raise_error=False)
        is_existed = True
        if diff_project_info is None:
            is_existed = False
            diff_project_info = self.api.project.create(
                diff_workspace_id, diff_project_name, change_name_if_conflict=True
            )
            for dataset in self.api.dataset.get_list(self.dt_project_info.id):
                self.api.dataset.create(diff_project_info.id, dataset.name)
        return diff_project_info, is_existed

    def upload_visualizations(self, dest_dir: str):
        layout_dir = self.get_layout_results_dir()
        assert not sly.fs.dir_empty(
            layout_dir
        ), f"The layout dir {layout_dir!r} is empty. You should run evaluation before uploading results."

        # self.api.file.remove_dir(self.team_id, dest_dir, silent=True)

        with tqdm_sly(
            desc="Uploading layout",
            total=get_directory_size(layout_dir),
            unit="B",
            unit_scale=True,
        ) as pbar:
            self.api.file.upload_directory(
                self.team_id,
                layout_dir,
                dest_dir,
                replace_if_conflict=True,
                change_name_if_conflict=False,
                progress_size_cb=pbar,
            )

        logger.info(f"Uploaded to: {dest_dir!r}")
