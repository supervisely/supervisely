# coding: utf-8

import os.path as osp
import os
import time
import shutil
# ''''''
# import supervisely_lib.worker_proto.worker_api_pb2 as api_proto
# from supervisely_lib import EventType
# from supervisely_lib.project.project_structure import ProjectFS
# from supervisely_lib.project.project_meta import ProjectMeta
# from supervisely_lib.utils.json_utils import json_dump
# from supervisely_lib.utils.os_utils import mkdir
#
# from src.worker.task_dockerized import TaskDockerized
# from src.worker.task_sly import TaskSly
# from src.worker.agent_utils import TaskDirCleaner
#
#
# exit_events = {0: EventType.TASK_FINISHED,
#                1: EventType.TASK_CRASHED,
#                2: EventType.TASK_STOPPED}
#
#
# def replace_annotations(src, scr_pr_name, dst, dst_pr_name):
#     project_src = ProjectFS.from_disk(src, scr_pr_name, by_annotations=True)
#     project_dst = ProjectFS.from_disk(dst, dst_pr_name)
#
#     if project_dst.image_cnt != project_src.image_cnt:
#         raise ValueError("replace_annotations will not work on different datasets")
#
#     for pr_item in project_dst:
#         new_ann_path = project_src.ann_path(pr_item.ds_name, pr_item.image_name)
#         old_ann_path = pr_item.ann_path
#         shutil.copy(new_ann_path, old_ann_path)
#
#     meta_src = ProjectMeta.from_dir(osp.join(src, scr_pr_name))
#     meta_dst = ProjectMeta.from_dir(osp.join(dst, dst_pr_name))
#     meta_dst.update(meta_src)
#     meta_dst.to_dir(osp.join(dst, dst_pr_name))
#
#
# class TaskDockerizedStep(TaskDockerized):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.docker_runtime = 'nvidia'
#
#     def clean_task_dir(self):
#         self.task_dir_cleaner.forbid_dir_cleaning()
#         pass
#
#     def before_main_step(self):
#         pass
#
#     def download_step(self):
#         pass
#
#     def upload_step(self):
#         pass
#
#     def end_log_stop(self):
#         return EventType.TASK_STOPPED
#
#     def end_log_crash(self, e):
#         return EventType.TASK_CRASHED
#
#     def end_log_finish(self):
#         return EventType.TASK_FINISHED
#
#
# class TaskPipeline(TaskSly):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.daemon = False
#
#         self.dir_data = osp.join(self.dir_task, 'data')
#         self.dir_results = osp.join(self.dir_task, 'results')
#         self.dir_model = osp.join(self.dir_task, 'model')
#         self.config_path = osp.join(self.dir_task, 'task_settings.json')
#         self.config_graph_path = osp.join(self.dir_task, 'graph.json')
#
#         self.stage_obj = None
#         self.task_dir_cleaner = TaskDirCleaner(self.dir_task)
#
#     def get_stage_model_dir(self, stage_name):
#         return '{}_{}'.format(self.dir_model, stage_name)
#
#     def get_stage_results_dir(self, stage_name):
#         return '{}_{}'.format(self.dir_results, stage_name)
#
#     def init_additional(self):
#         super().init_additional()
#
#         # download all nn weights
#
#         for nn_info in self.info.get('models', []):
#             self.data_mgr.download_nn(nn_info['title'], nn_info['id'], nn_info['hash'], self.get_stage_model_dir(nn_info['title']))
#
#         for stage_config in self.info['config']['stages']:
#             stage_nn_info = self.data_mgr.get_nn_info_by_name(stage_config['nn_model'])
#             stage_config['docker_image'] = stage_nn_info.arch.inf_docker
#
#         # download input data from server
#         if len(self.info['projects']) != 1:
#             raise ValueError("Config contains {} projects. Training works only with single project.".format(len(self.info['projects'])))
#         self.info['project'] = self.info['projects'][0]
#         pr_info = self.info['project']
#
#         frame_step = self.info['config'].get('frame_step', 1)
#         frame_range = self.info['config'].get('frame_range')
#
#         project = api_proto.Project(id=pr_info['id'], title=pr_info['title'])
#         datasets = [api_proto.Dataset(id=ds['id'], title=ds['title']) for ds in pr_info['datasets']]
#         self.data_mgr.download_project(self.dir_data, project, datasets, True, frame_step, frame_range)
#
#
#     # def download_stage_nn(self, stage_config, stage_model_dir):
#     #     nn_name = stage_config.get('nn_model', None)
#     #     if nn_name  is None:
#     #         self.logger.critical('TASK_NN_EMPTY', extra={'stage_name': stage_config['stage_name']})
#     #         raise ValueError('TASK_NN_EMPTY')
#     #
#     #     nn_info = self.data_mgr.get_nn_info_by_name(nn_name)
#     #     nn_id = nn_info.desc.id
#     #     nn_hash = nn_info.desc.hash
#     #
#     #     self.logger.info('DOWNLOAD_NN', extra={'nn_id': nn_id, 'nn_hash': nn_hash})
#     #     self.data_mgr.download_nn(nn_id, nn_hash, stage_model_dir)
#     #
#     #     return nn_info
#
#     def join(self, timeout=None):
#         if self.stage_obj is not None:
#             self.stage_obj.join(timeout)
#
#         super().join(timeout)
#
#     def clean_task_dir(self):
#         self.task_dir_cleaner.clean()
#
#     def upload_project(self, result_project_pathname, project_title):
#         pr_id = self.data_mgr.upload_project(result_project_pathname,
#                                              project_title,
#                                              no_image_files=True,
#                                              create_new_project=True, recalculate_images_hash=False)
#         self.logger.info('PROJECT_CREATED', extra={'event_type': EventType.PROJECT_CREATED, 'project_id': pr_id})
#
#     def task_main_func(self):
#         self.task_dir_cleaner.forbid_dir_cleaning()
#
#         postprocessing_stage = self.info['config'].get('postprocessing')
#         if postprocessing_stage is not None:
#             self.info['config']['stages'].append(postprocessing_stage)
#
#         stages = self.info['config']['stages']
#         for step_idx, stage_info in enumerate(stages):
#             is_postprocessing_stage = stage_info == postprocessing_stage
#             stage_name = stage_info['name']
#             self.logger.info('Stage is started', extra={'stage_name': stage_name})
#
#             stage_info['task_id'] = self.info['task_id']
#
#             if is_postprocessing_stage:
#                 for node in stage_info['graph']:
#                     if node['action'] == 'data':
#                         node['src'] = [self.info['project']['title'] + '/*']
#                 json_dump(stage_info['graph'], self.config_graph_path)
#                 stage_entrypoint = "/workdir/src/main.py"
#             else:
#                 json_dump(stage_info['config'], self.config_path)
#                 os.rename(self.get_stage_model_dir(stage_info['nn_model']), self.dir_model)
#                 stage_entrypoint = "/workdir/src/inference.py"
#
#             mkdir(self.dir_results)
#
#             stage_obj = TaskDockerizedStep(stage_info)
#             stage_obj.docker_api = self.docker_api
#             stage_obj.entrypoint = stage_entrypoint
#
#             stage_obj.start()
#             while stage_obj.is_alive():
#                 time.sleep(1)
#
#             stage_exit_event = exit_events[stage_obj.exitcode]
#             if stage_exit_event is EventType.TASK_CRASHED:
#                 raise ValueError("STAGE CRASHED")
#             elif stage_exit_event is EventType.TASK_STOPPED:
#                 return
#
#             if not is_postprocessing_stage:
#                 # Upload project before postprocessing stage
#                 is_next_postprocessing_stage = False
#                 next_idx = step_idx+1
#                 if next_idx < len(stages):
#                     is_next_postprocessing_stage = stages[next_idx] == postprocessing_stage
#                 if next_idx == len(stages) or is_next_postprocessing_stage:
#                     result_project_pathname = osp.join(self.dir_results, self.info['project']['title'])
#                     self.upload_project(result_project_pathname, project_title=self.info['new_title'])
#
#                 os.rename(self.dir_model, self.get_stage_model_dir(stage_info['nn_model']))
#                 os.rename(self.dir_results, self.get_stage_results_dir(stage_name))
#                 replace_annotations(
#                     self.get_stage_results_dir(stage_name),
#                     self.info['project']['title'],
#                     self.dir_data,
#                     self.info['project']['title'])
#
#             self.logger.info('Stage is finished', extra={'stage_name': stage_name})
#
#         # Upload second project after postprocessing stage
#         if postprocessing_stage is not None:
#             title = self.info['new_title']
#             new_title = title + '_postprocessed'
#             self.upload_project(result_project_pathname=self.dir_results, project_title=new_title)
#
#         self.task_dir_cleaner.allow_cleaning()