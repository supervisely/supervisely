# coding: utf-8

import os.path as osp

import supervisely_lib as sly
from supervisely_lib import EventType
import supervisely_lib.worker_proto as api_proto

from .task_dockerized import TaskDockerized, TaskStep
from .agent_utils import ann_special_fields


class TaskDTL(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_map = {
            str(EventType.DTL_APPLIED): self.upload_result_project,
            str(EventType.TASK_VERIFIED): self.on_verify
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
        res = sly.json_load(self.verif_status_path)
        self.download_images = res['download_images']
        self.is_archive = res['is_archive']

    def init_additional(self):
        super().init_additional()
        sly.mkdir(self.dir_data)
        sly.mkdir(self.dir_results)
        sly.json_dump(self.info['graph'], self.graph_path)

    def download_data_sources(self, only_meta=False):
        self.logger.info("download_data_sources started")
        data_sources = sly.get_data_sources(self.info['graph'])
        for proj, datasets in data_sources.items():
            pr_name = proj
            pr_proto = self.api.simple_request('GetProjectByName', api_proto.Project, api_proto.Project(title=pr_name))
            if pr_proto.id == -1:
                self.logger.critical('Project not found', extra={'project_name': pr_name})
                raise RuntimeError('Project not found')

            datasets_proto_arr = []
            if datasets != "*":
                for ds_name in datasets:
                    ds_proto = self.api.simple_request(
                        'GetDatasetByName',
                        api_proto.Dataset,
                        api_proto.ProjectDataset(project=api_proto.Project(id=pr_proto.id),
                                                 dataset=api_proto.Dataset(title=ds_name)))
                    if ds_proto.id == -1:
                        self.logger.critical('Dataset not found', extra={'project_id': pr_proto.id,
                                                                         'project_title': pr_name,
                                                                         'dataset_title': ds_name})
                        raise RuntimeError('Dataset not found')
                    datasets_proto_arr.append(api_proto.Dataset(id=ds_proto.id, title=ds_name))
            else:
                datasets_proto = self.api.simple_request('GetProjectDatasets',
                                                         api_proto.DatasetArray,
                                                         api_proto.Id(id=pr_proto.id))
                datasets_proto_arr = datasets_proto.datasets

            if only_meta is True:
                project_info = self.api.simple_request('GetProjectMeta',
                                                       api_proto.Project,
                                                       api_proto.Id(id=pr_proto.id))
                pr_writer = sly.ProjectWriterFS(self.dir_data, project_info.title)
                pr_meta = sly.ProjectMeta(sly.json_loads(project_info.meta))
                pr_writer.write_meta(pr_meta)
            else:
                self.data_mgr.download_project(self.dir_data, pr_proto, datasets_proto_arr,
                                               download_images=self.download_images)

    def verify(self):
        self.download_data_sources(only_meta=True)
        self.docker_pull_if_needed()
        self.spawn_container(add_envs={'VERIFICATION': '1'})
        self.process_logs()
        self.drop_container_and_check_status()

        self.logger.info('VERIFY_END')
        sly.clean_dir(self.dir_data)
        self._read_verif_status()

    def download_step(self):
        self.verify()
        self.logger.info("DOWNLOAD_DATA")
        self.download_data_sources()
        self.report_step_done(TaskStep.DOWNLOAD)

    def on_verify(self, jlog):
        download_images = jlog.get('output', {}).get('download_images', None)
        is_archive = jlog.get('output', {}).get('is_archive', None)
        if download_images is None or is_archive is None:
            raise ValueError('VERIFY_IS_NONE')
        sly.json_dump(
            {'download_images': download_images, 'is_archive': is_archive},
            self.verif_status_path
        )
        return {}

    def before_main_step(self):
        sly.clean_dir(self.dir_results)

    def upload_step(self):
        self.upload_result_project({})

    def upload_result_project(self, _):
        self.report_step_done(TaskStep.MAIN)
        self._read_verif_status()
        graph_pr_name = sly.get_res_project_name(self.info['graph'])

        no_image_files = not self.download_images

        if self.is_archive is False:
            pr_id = self.data_mgr.upload_project(self.dir_results, graph_pr_name, no_image_files=no_image_files)
            self.logger.info('PROJECT_CREATED', extra={'event_type': EventType.PROJECT_CREATED, 'project_id': pr_id})
        else:
            # remove excess fields from json
            root_path, project_name = sly.ProjectFS.split_dir_project(self.dir_results)
            project_fs = sly.ProjectFS.from_disk(root_path, project_name, by_annotations=True)
            for item in project_fs:
                ann_path = item.ann_path
                ann = sly.json_load(ann_path)
                for exc_field in ann_special_fields():
                    ann.pop(exc_field, None)
                sly.json_dump(ann, ann_path)

            self.data_mgr.upload_archive(self.dir_results, graph_pr_name)

        self.report_step_done(TaskStep.UPLOAD)
        return {}
