# coding: utf-8

import os.path as osp
import os
import json
from collections import defaultdict
import supervisely_lib as sly

from worker.task_dockerized import TaskDockerized, TaskStep
from worker.agent_utils import ann_special_fields
from supervisely_lib.io.json import dump_json_file


class TaskDTL(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_map = {
            str(sly.EventType.DTL_APPLIED): self.upload_result_project,
            str(sly.EventType.TASK_VERIFIED): self.on_verify
        }

        self.download_images = None
        self.is_archive = None

        self.dir_data = osp.join(self.dir_task, 'data')
        self.graph_path = osp.join(self.dir_task, 'graph.json')
        self.dir_results = osp.join(self.dir_task, 'results')
        self.verif_status_path = osp.join(self.dir_task, 'verified.json')

    def _read_verif_status(self):
        if not osp.isfile(self.verif_status_path):
            raise RuntimeError('VERIFY_FAILED')
        res = json.load(open(self.verif_status_path, 'r'))
        self.is_archive = res['is_archive']

    def init_additional(self):
        super().init_additional()
        sly.fs.mkdir(self.dir_data)
        sly.fs.mkdir(self.dir_results)
        dump_json_file(self.info['graph'], self.graph_path)

    def download_data_sources(self, only_meta=False):
        data_sources = _get_data_sources(self.info['graph'])
        for project_name, datasets in data_sources.items():
            project_id = self.public_api.project.get_info_by_name(self.data_mgr.workspace_id, project_name).id
            if only_meta is True:
                meta_json = self.public_api.project.get_meta(project_id)
                project = sly.Project(os.path.join(self.dir_data, project_name), sly.OpenMode.CREATE)
                project.set_meta(sly.ProjectMeta.from_json(meta_json))
            else:
                datasets_to_download = None  #will download all datasets
                if datasets != "*":
                    datasets_to_download = datasets
                self.data_mgr.download_project(self.dir_data, project_name, datasets_to_download)

    def verify(self):
        self.download_data_sources(only_meta=True)
        self.spawn_container(add_envs={'VERIFICATION': '1'})
        self.process_logs()
        self.drop_container_and_check_status()

        self.logger.info('VERIFY_END')
        sly.fs.clean_dir(self.dir_data)
        self._read_verif_status()

    def download_step(self):
        self.verify()
        self.logger.info("DOWNLOAD_DATA")
        self.download_data_sources()
        self.report_step_done(TaskStep.DOWNLOAD)

    def on_verify(self, jlog):
        is_archive = jlog['output']['is_archive']
        dump_json_file({'is_archive': is_archive}, self.verif_status_path)
        return {}

    def before_main_step(self):
        sly.fs.clean_dir(self.dir_results)

    def upload_step(self):
        self.upload_result_project({})

    def upload_result_project(self, _):
        self.report_step_done(TaskStep.MAIN)
        self._read_verif_status()

        graph_pr_name = _get_res_project_name(self.info['graph'])
        if self.is_archive is False:
            self.data_mgr.upload_project(self.dir_results, graph_pr_name, graph_pr_name, legacy=True)
        else:
            self.data_mgr.upload_archive(self.info['task_id'], self.dir_results, graph_pr_name)

        self.report_step_done(TaskStep.UPLOAD)
        return {}


def _get_data_sources(graph_json):
    all_ds_marker = '*'
    data_sources = defaultdict(set)
    for layer in graph_json:
        if layer['action'] == 'data':
            for src in layer['src']:
                src_parts = src.split('/')
                src_project_name = src_parts[0]
                src_dataset_name = src_parts[1] if len(src_parts) > 1 else all_ds_marker
                data_sources[src_project_name].add(src_dataset_name)

    def _squeeze_datasets(datasets):
        if all_ds_marker in datasets:
            res = all_ds_marker
        else:
            res = list(datasets)
        return res

    data_sources = {k: _squeeze_datasets(v) for k, v in data_sources.items()}
    return data_sources


def _get_res_project_name(graph_json):
    for layer in graph_json:
        if layer['action'] in ['supervisely', 'save', 'save_masks']:
            return layer['dst']
    raise RuntimeError('Supervisely save layer not found.')