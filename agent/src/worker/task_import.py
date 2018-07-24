# coding: utf-8

import os.path as osp

import pyexiv2
import supervisely_lib as sly
from supervisely_lib import EventType

from .task_dockerized import TaskDockerized, TaskStep


class TaskImport(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_map = {
            str(EventType.IMPORT_APPLIED): self.upload_result_project
        }

        self.human_config = {'res_names': {'project': self.info['project_name']}, 'preset': self.info['preset']}
        self.dir_data = osp.join(self.dir_task, 'data')
        self.dir_results = osp.join(self.dir_task, 'results')
        self.config_path = osp.join(self.dir_task, 'task_settings.json')
        self.dir_res_project = osp.join(self.dir_results, self.info['project_name'])

    def init_additional(self):
        super().init_additional()
        sly.mkdir(self.dir_data)
        sly.mkdir(self.dir_results)

    def download_step(self):
        self.logger.info("DOWNLOAD_DATA")
        sly.json_dump(self.human_config, self.config_path)
        self.data_mgr.download_import_files(self.info['task_id'], self.dir_data)
        self.report_step_done(TaskStep.DOWNLOAD)

    def before_main_step(self):
        sly.clean_dir(self.dir_results)

    def upload_step(self):
        self.upload_result_project({})

    def _exif_checking_step(self):
        self.logger.info('Check orientation of images')

        root_path, project_name = sly.ProjectFS.split_dir_project(self.dir_res_project)
        project_fs = sly.ProjectFS.from_disk(root_path, project_name, by_annotations=True)

        progress = sly.ProgressCounter('EXIF checking', project_fs.image_cnt, ext_logger=self.logger)

        for descr in project_fs:
            img_path = descr.img_path
            exif_data = pyexiv2.metadata.ImageMetadata(img_path)
            exif_data.read()

            if exif_data.get_orientation() != 1:
                raise RuntimeError('Wrong image orientation in EXIF. Image name: {}'.format(osp.basename(img_path)))

            progress.iter_done_report()

    def upload_result_project(self, _):
        self.report_step_done(TaskStep.MAIN)
        self._exif_checking_step()
        pr_id = self.data_mgr.upload_project(self.dir_res_project, self.info['project_name'], no_image_files=False)
        self.logger.info('PROJECT_CREATED', extra={'event_type': EventType.PROJECT_CREATED, 'project_id': pr_id})
        self.report_step_done(TaskStep.UPLOAD)
        return {}
