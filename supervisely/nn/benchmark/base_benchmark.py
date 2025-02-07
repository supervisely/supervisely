import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np

from supervisely._utils import is_development
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import SlyTqdm
from supervisely.io import env, fs, json
from supervisely.io.fs import get_directory_size
from supervisely.nn.benchmark.base_evaluator import BaseEvaluator
from supervisely.nn.inference import SessionJSON
from supervisely.project.project import download_project
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly

WORKSPACE_NAME = "Model Benchmark: predictions and differences"
WORKSPACE_DESCRIPTION = "Technical workspace for model benchmarking. Contains predictions and differences between ground truth and predictions."


class BaseBenchmark:
    visualizer_cls = None
    EVALUATION_DIR_NAME = "evaluation"
    SPEEDTEST_DIR_NAME = "speedtest"
    VISUALIZATIONS_DIR_NAME = "visualizations"

    def __init__(
        self,
        api: Api,
        gt_project_id: int,
        gt_dataset_ids: List[int] = None,
        gt_images_ids: List[int] = None,
        output_dir: str = "./benchmark",
        progress: Optional[SlyTqdm] = None,
        progress_secondary: Optional[SlyTqdm] = None,
        classes_whitelist: Optional[List[str]] = None,
        evaluation_params: Optional[dict] = None,
    ):
        self.api = api
        self.session: SessionJSON = None
        self.gt_project_info = api.project.get_info_by_id(gt_project_id)
        self.dt_project_info: ProjectInfo = None
        self.diff_project_info: ProjectInfo = None
        self.gt_dataset_ids = gt_dataset_ids
        self.gt_images_ids = gt_images_ids
        self.gt_dataset_infos = None
        if gt_dataset_ids is not None:
            self.gt_dataset_infos = self.api.dataset.get_list(
                self.gt_project_info.id,
                filters=[
                    {
                        ApiField.FIELD: ApiField.ID,
                        ApiField.OPERATOR: "in",
                        ApiField.VALUE: gt_dataset_ids,
                    }
                ],
                recursive=True,
            )
        self.num_items = self._get_total_items_for_progress()
        self.output_dir = output_dir
        self.team_id = env.team_id()
        self.evaluator: BaseEvaluator = None
        self._eval_inference_info = None
        self._speedtest = None
        self._hardware = None
        self.pbar = progress or tqdm_sly
        self.progress_secondary = progress_secondary or tqdm_sly
        self.classes_whitelist = classes_whitelist
        self.vis_texts = None
        self.inference_speed_text = None
        self.train_info = None
        self.evaluator_app_info = None
        self.evaluation_params = evaluation_params
        self.visualizer = None
        self.remote_vis_dir = None
        self._eval_results = None
        self.report_id = None
        self._validate_evaluation_params()

    def _get_evaluator_class(self) -> Type[BaseEvaluator]:
        raise NotImplementedError()

    @property
    def cv_task(self) -> str:
        raise NotImplementedError()

    @property
    def hardware(self) -> str:
        return self._hardware

    @property
    def key_metrics(self):
        eval_results = self.get_eval_result()
        return eval_results.key_metrics

    @property
    def primary_metric_name(self) -> str:
        return self._get_evaluator_class().eval_result_cls.PRIMARY_METRIC

    def run_evaluation(
        self,
        model_session: Union[int, str, SessionJSON],
        inference_settings=None,
        output_project_id=None,
        batch_size: int = 16,
        cache_project_on_agent: bool = False,
    ):
        self.run_inference(
            model_session=model_session,
            inference_settings=inference_settings,
            output_project_id=output_project_id,
            batch_size=batch_size,
            cache_project_on_agent=cache_project_on_agent,
        )
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
        self._eval_inference_info = self._run_inference(
            output_project_id, batch_size, cache_project_on_agent
        )

    def _run_inference(
        self,
        output_project_id=None,
        batch_size: int = 16,
        cache_project_on_agent: bool = False,
    ):
        model_info = self._fetch_model_info(self.session)
        self.dt_project_info = self._get_or_create_dt_project(output_project_id, model_info)
        logger.info(
            f"""
            Predictions project ID: {self.dt_project_info.id},
                        workspace ID: {self.dt_project_info.workspace_id}"""
        )
        if self.gt_images_ids is None:
            iterator = self.session.inference_project_id_async(
                self.gt_project_info.id,
                self.gt_dataset_ids,
                output_project_id=self.dt_project_info.id,
                cache_project_on_model=cache_project_on_agent,
                batch_size=batch_size,
            )
        else:
            iterator = self.session.inference_image_ids_async(
                image_ids=self.gt_images_ids,
                output_project_id=self.dt_project_info.id,
                batch_size=batch_size,
            )
        output_project_id = self.dt_project_info.id
        with self.pbar(message="Evaluation: Running inference", total=self.num_items) as p:
            for _ in iterator:
                p.update(1)
        inference_info = {
            "gt_project_id": self.gt_project_info.id,
            "gt_dataset_ids": self.gt_dataset_ids,
            "dt_project_id": output_project_id,
            "batch_size": batch_size,
            **model_info,
        }
        if self.train_info:
            self.train_info.pop("train_images_ids", None)
            inference_info["train_info"] = self.train_info
        if self.evaluator_app_info:
            self.evaluator_app_info.pop("settings", None)
            inference_info["evaluator_app_info"] = self.evaluator_app_info
        if self.gt_images_ids:
            inference_info["val_images_cnt"] = len(self.gt_images_ids)
        self.dt_project_info = self.api.project.get_info_by_id(self.dt_project_info.id)
        logger.debug(
            "Inference is finished.",
            extra={
                "inference_info": inference_info,
                "dt_project_info": self.dt_project_info._asdict(),
            },
        )

        self._merge_metas(self.gt_project_info.id, self.dt_project_info.id)
        return inference_info

    def evaluate(self, dt_project_id):
        self.dt_project_info = self.api.project.get_info_by_id(dt_project_id)
        gt_project_path, dt_project_path = self._download_projects()
        self._evaluate(gt_project_path, dt_project_path)

    def _evaluate(self, gt_project_path, dt_project_path):
        eval_results_dir = self.get_eval_results_dir()
        self.evaluator = self._get_evaluator_class()(
            gt_project_path=gt_project_path,
            pred_project_path=dt_project_path,
            result_dir=eval_results_dir,
            progress=self.pbar,
            items_count=self.num_items,
            classes_whitelist=self.classes_whitelist,
            evaluation_params=self.evaluation_params,
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
        self._hardware = model_info["hardware"]
        benchmarks = []
        with self.pbar(
            message="Speedtest: Running speedtest for batch sizes", total=len(batch_sizes)
        ) as p:
            for bs in batch_sizes:
                logger.debug(f"Running speedtest for batch_size={bs}")
                speedtest_results = []
                iterator = self.session.run_speedtest(
                    project_id,
                    batch_size=bs,
                    num_iterations=num_iterations,
                    num_warmup=num_warmup,
                    preparing_cb=self.progress_secondary,
                    cache_project_on_model=cache_project_on_agent,
                )
                p2 = self.progress_secondary(
                    message="Speedtest: Running inference", total=len(iterator)
                )
                for speedtest in iterator:
                    speedtest_results.append(speedtest)
                    p2.update(1)
                assert (
                    len(speedtest_results) == num_iterations
                ), "Speedtest failed to run all iterations."
                logger.info(f"Inference finished for batch_size={bs}. Calculating statistics.")
                avg_speedtest, std_speedtest = self._calculate_speedtest_statistics(
                    speedtest_results
                )
                benchmark = {
                    "benchmark": avg_speedtest,
                    "benchmark_std": std_speedtest,
                    "batch_size": bs,
                    **speedtest_info,
                }
                benchmarks.append(benchmark)
                p.update(1)
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
        dt_path = os.path.join(base_dir, "pred_project")
        return gt_path, dt_path

    def get_eval_results_dir(self) -> str:
        eval_dir = os.path.join(self.get_base_dir(), self.EVALUATION_DIR_NAME)
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir

    def get_speedtest_results_dir(self) -> str:
        speedtest_dir = os.path.join(self.get_base_dir(), self.SPEEDTEST_DIR_NAME)
        os.makedirs(speedtest_dir, exist_ok=True)
        return speedtest_dir

    def upload_eval_results(self, remote_dir: str):
        eval_dir = self.get_eval_results_dir()
        assert not fs.dir_empty(
            eval_dir
        ), f"The result dir {eval_dir!r} is empty. You should run evaluation before uploading results."
        with self.pbar(
            message="Evaluation: Uploading evaluation results",
            total=fs.get_directory_size(eval_dir),
            unit="B",
            unit_scale=True,
        ) as p:
            self.api.file.upload_directory(
                self.team_id,
                eval_dir,
                remote_dir,
                replace_if_conflict=True,
                change_name_if_conflict=False,
                progress_size_cb=p,
            )

    def get_layout_results_dir(self) -> str:
        dir = os.path.join(self.get_base_dir(), self.VISUALIZATIONS_DIR_NAME)
        os.makedirs(dir, exist_ok=True)
        return dir

    def upload_speedtest_results(self, remote_dir: str):
        speedtest_dir = self.get_speedtest_results_dir()
        assert not fs.dir_empty(
            speedtest_dir
        ), f"Speedtest dir {speedtest_dir!r} is empty. You should run speedtest before uploading results."
        self.api.file.upload_directory(self.team_id, speedtest_dir, remote_dir)

    def _generate_dt_project_name(self, gt_project_name, model_info):
        dt_project_name = gt_project_name + " - " + model_info["checkpoint_name"]
        return dt_project_name

    def _generate_diff_project_name(self, dt_project_name):
        return "[diff]: " + dt_project_name

    def _get_or_create_dt_project(self, output_project_id, model_info) -> ProjectInfo:
        if output_project_id is None:
            dt_project_name = self._generate_dt_project_name(self.gt_project_info.name, model_info)
            workspace = self.api.workspace.get_info_by_name(self.team_id, WORKSPACE_NAME)
            if workspace is None:
                workspace = self.api.workspace.create(
                    self.team_id, WORKSPACE_NAME, WORKSPACE_DESCRIPTION
                )
            visible = is_development()
            self.api.workspace.change_visibility(workspace.id, visible=visible)
            dt_project_info = self.api.project.create(
                workspace.id, dt_project_name, change_name_if_conflict=True
            )
            output_project_id = dt_project_info.id
        else:
            dt_project_info = self.api.project.get_info_by_id(output_project_id)
        return dt_project_info

    def download_projects(self, save_images: bool = False):
        return self._download_projects(save_images=save_images)

    def _download_projects(self, save_images=False):
        gt_path, dt_path = self.get_project_paths()
        if not os.path.exists(gt_path):
            with self.pbar(
                message="Evaluation: Downloading GT annotations", total=self.num_items
            ) as p:
                download_project(
                    self.api,
                    self.gt_project_info.id,
                    gt_path,
                    dataset_ids=self.gt_dataset_ids,
                    log_progress=True,
                    save_images=save_images,
                    save_image_info=True,
                    progress_cb=p.update,
                    images_ids=self.gt_images_ids,
                )
        else:
            logger.info(f"Found GT annotations in {gt_path}")
        if not os.path.exists(dt_path):
            with self.pbar(
                message="Evaluation: Downloading prediction annotations", total=self.num_items
            ) as p:
                download_project(
                    self.api,
                    self.dt_project_info.id,
                    dt_path,
                    log_progress=True,
                    save_images=save_images,
                    save_image_info=True,
                    progress_cb=p.update,
                )
        else:
            logger.info(f"Found Pred annotations in {dt_path}")

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
            app_info = None
        model_info = {
            **deploy_info,
            "inference_settings": session.inference_settings,
            "app_info": app_info,
        }
        return model_info

    def _init_model_session(
        self, model_session: Union[int, str, SessionJSON], inference_settings: dict = None
    ):
        if isinstance(model_session, int):
            session = SessionJSON(self.api, model_session)
        elif isinstance(model_session, str):
            session = SessionJSON(self.api, session_url=model_session)
        elif isinstance(model_session, SessionJSON):
            session = model_session
        else:
            raise ValueError(f"Unsupported type of 'model_session' argument: {type(model_session)}")

        if self.classes_whitelist:
            inference_settings = inference_settings or {}
            inference_settings["classes"] = self.classes_whitelist

        if inference_settings is not None:
            session.set_inference_settings(inference_settings)
        return session

    def _dump_project_info(self, project_info: ProjectInfo, project_path):
        project_info_path = os.path.join(project_path, "project_info.json")
        json.dump_json_file(project_info._asdict(), project_info_path, indent=2)
        return project_info_path

    def _dump_eval_inference_info(self, eval_inference_info):
        info_path = os.path.join(self.get_eval_results_dir(), "inference_info.json")
        json.dump_json_file(eval_inference_info, info_path)
        return info_path

    def _dump_speedtest(self, speedtest):
        path = os.path.join(self.get_speedtest_results_dir(), "speedtest.json")
        json.dump_json_file(speedtest, path, indent=2)
        return path

    def _calculate_speedtest_statistics(self, speedtest_results: list):
        x = [[s[k] for s in speedtest_results] for k in speedtest_results[0].keys()]
        x = np.array(x, dtype=float)
        avg = x.mean(1)
        std = x.std(1)
        avg_speedtest = {
            k: float(avg[i]) if not np.isnan(avg[i]).any() else None
            for i, k in enumerate(speedtest_results[0].keys())
        }
        std_speedtest = {
            k: float(std[i]) if not np.isnan(std[i]).any() else None
            for i, k in enumerate(speedtest_results[0].keys())
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
        if dt_project_id is not None:
            self.dt_project_info = self.api.project.get_info_by_id(dt_project_id)

        if self.visualizer_cls is None:
            raise RuntimeError(
                f"Visualizer class is not defined in {self.__class__.__name__}. "
                "It should be defined in the subclass of BaseBenchmark (e.g. ObjectDetectionBenchmark)."
            )
        eval_result = self.get_eval_result()
        self._dump_key_metrics(eval_result)

        layout_dir = self.get_layout_results_dir()
        self.visualizer = self.visualizer_cls(  # pylint: disable=not-callable
            self.api, [eval_result], layout_dir, self.pbar
        )
        with self.pbar(message="Visualizations: Rendering layout", total=1) as p:
            self.visualizer.visualize()
            p.update(1)

    def _get_or_create_diff_project(self) -> Tuple[ProjectInfo, bool]:

        dt_ds_id_to_diff_ds_info = {}

        def _get_or_create_diff_dataset(dt_dataset_id, dt_datasets):
            if dt_dataset_id in dt_ds_id_to_diff_ds_info:
                return dt_ds_id_to_diff_ds_info[dt_dataset_id]
            dt_dataset = dt_datasets[dt_dataset_id]
            if dt_dataset.parent_id is None:
                diff_dataset = self.api.dataset.create(
                    diff_project_info.id,
                    dt_dataset.name,
                )
            else:
                parent_dataset = _get_or_create_diff_dataset(dt_dataset.parent_id, dt_datasets)
                diff_dataset = self.api.dataset.create(
                    diff_project_info.id,
                    dt_dataset.name,
                    parent_id=parent_dataset.id,
                )
            dt_ds_id_to_diff_ds_info[dt_dataset_id] = diff_dataset
            return diff_dataset

        diff_project_name = self._generate_diff_project_name(self.dt_project_info.name)
        diff_workspace_id = self.dt_project_info.workspace_id
        diff_project_info = self.api.project.get_info_by_name(
            diff_workspace_id, diff_project_name, raise_error=False
        )
        is_existed = True
        if diff_project_info is None:
            is_existed = False
            diff_project_info = self.api.project.create(
                diff_workspace_id, diff_project_name, change_name_if_conflict=True
            )
            dt_datasets = {
                ds.id: ds
                for ds in self.api.dataset.get_list(self.dt_project_info.id, recursive=True)
            }
            for dataset in dt_datasets:
                _get_or_create_diff_dataset(dataset, dt_datasets)
        return diff_project_info, is_existed

    def upload_visualizations(self, dest_dir: str):
        self.remote_vis_dir = self.visualizer.upload_results(self.team_id, dest_dir, self.pbar)
        return self.remote_vis_dir

    @property
    def report(self):
        return self.visualizer.renderer.report

    @property
    def lnk(self):
        return self.visualizer.renderer.lnk

    def upload_report_link(self, remote_dir: str, report_id: int = None, local_dir: str = None):
        if report_id is None:
            template_path = os.path.join(remote_dir, "template.vue")
            vue_template_info = self.api.file.get_info_by_path(self.team_id, template_path)
            self.report_id = vue_template_info.id
            report_id = vue_template_info.id

        report_link = "/model-benchmark?id=" + str(report_id)
        lnk_name = "Model Evaluation Report.lnk"

        if local_dir is None:
            local_dir = self.get_layout_results_dir()
        local_path = os.path.join(local_dir, lnk_name)

        with open(local_path, "w") as file:
            file.write(report_link)

        remote_path = os.path.join(remote_dir, lnk_name)
        file_info = self.api.file.upload(self.team_id, local_path, remote_path)

        logger.info(f"Report link: {report_link}")
        return file_info

    def get_report_link(self) -> str:
        if self.remote_vis_dir is None:
            raise ValueError("Visualizations are not uploaded yet.")
        return self.visualizer.renderer._get_report_link(
            self.api, self.team_id, self.remote_vis_dir
        )

    def _merge_metas(self, gt_project_id, pred_project_id):
        gt_meta = self.api.project.get_meta(gt_project_id)
        gt_meta = ProjectMeta.from_json(gt_meta)

        pred_meta = self.api.project.get_meta(pred_project_id)
        pred_meta = ProjectMeta.from_json(pred_meta)

        chagned = False
        for obj_cls in gt_meta.obj_classes:
            if not pred_meta.obj_classes.has_key(obj_cls.name):
                pred_meta = pred_meta.add_obj_class(obj_cls)
                chagned = True
        for tag_meta in gt_meta.tag_metas:
            if not pred_meta.tag_metas.has_key(tag_meta.name):
                pred_meta = pred_meta.add_tag_meta(tag_meta)
                chagned = True
        if chagned:
            self.api.project.update_meta(pred_project_id, pred_meta.to_json())

    def _validate_evaluation_params(self):
        if self.evaluation_params:
            self._get_evaluator_class().validate_evaluation_params(self.evaluation_params)

    def _get_total_items_for_progress(self):
        if self.gt_images_ids is not None:
            return len(self.gt_images_ids)
        elif self.gt_dataset_ids is not None:
            return sum(ds.items_count for ds in self.gt_dataset_infos)
        else:
            return self.gt_project_info.items_count

    def get_eval_result(self):
        if self._eval_results is None:
            self._eval_results = self.evaluator.get_eval_result()
            if not self._eval_results.inference_info:
                self._eval_results.inference_info["gt_project_id"] = self.gt_project_info.id
                self._eval_results.inference_info["dt_project_id"] = self.dt_project_info.id
        return self._eval_results

    def get_diff_project_info(self):
        eval_result = self.get_eval_result()
        if hasattr(eval_result, "diff_project_info"):
            self.diff_project_info = eval_result.diff_project_info
            return self.diff_project_info
        return None

    def _dump_key_metrics(self, eval_result: BaseEvaluator):
        path = str(Path(self.get_eval_results_dir(), "key_metrics.json"))
        json.dump_json_file(eval_result.key_metrics, path)
