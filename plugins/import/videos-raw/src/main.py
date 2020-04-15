# coding: utf-8

from supervisely_lib.io.json import load_json_file
from supervisely_lib import TaskPaths
import supervisely_lib as sly
from supervisely_lib.video.video import _check_video_requires_processing, _SUPPORTED_CODECS, _SUPPORTED_CONTAINERS, \
                                        get_video_streams, warn_video_requires_processing, gen_video_stream_name
from supervisely_lib.video.import_utils import get_dataset_name


DEFAULT_DATASET_NAME = 'ds0'
root_ds_name = DEFAULT_DATASET_NAME


def add_videos_to_project():
    task_config = load_json_file(TaskPaths.TASK_CONFIG_PATH)

    task_id = task_config['task_id']
    append_to_existing_project = task_config['append_to_existing_project']

    server_address = task_config['server_address']
    token = task_config['api_token']

    instance_type = task_config.get("instance_type", sly.ENTERPRISE)

    api = sly.Api(server_address, token)

    task_info = api.task.get_info_by_id(task_id)
    api.add_additional_field('taskId', task_id)
    api.add_header('x-task-id', str(task_id))

    workspace_id = task_info["workspaceId"]
    project_name = task_config['project_name']

    project_info = None
    if append_to_existing_project is True:
        project_info = api.project.get_info_by_name(workspace_id, project_name, expected_type=sly.ProjectType.VIDEOS, raise_error=True)

    files_list = api.task.get_import_files_list(task_id)
    for file_info in files_list:
        original_path = file_info["filename"]
        try:
            file_name = sly.fs.get_file_name_with_ext(original_path)
            all_streams = file_info["meta"]["streams"]
            hash = file_info["hash"]
            ds_info = None

            video_streams = get_video_streams(all_streams)
            for stream_info in video_streams:
                stream_index = stream_info["index"]

                if instance_type == sly.COMMUNITY:
                    if _check_video_requires_processing(file_info, stream_info) is True:
                        warn_video_requires_processing(file_name)
                        continue

                if project_info is None:
                    project_info = api.project.create(workspace_id, project_name, type=sly.ProjectType.VIDEOS, change_name_if_conflict=True)

                if ds_info is None:
                    ds_name = get_dataset_name(original_path)
                    ds_info = api.dataset.get_or_create(project_info.id, ds_name)

                item_name = file_name
                info = api.video.get_info_by_name(ds_info.id, item_name)
                if info is not None:
                    item_name = gen_video_stream_name(file_name, stream_index)
                    sly.logger.warning("Name {!r} already exists in dataset {!r}: renamed to {!r}"
                                       .format(file_name, ds_info.name, item_name))

                _ = api.video.upload_hash(ds_info.id, item_name, hash, stream_index)
        except Exception as e:
            sly.logger.warning("File skipped {!r}: error occurred during processing {!r}".format(original_path, str(e)))

    if project_info is not None:
        sly.logger.info('PROJECT_CREATED', extra={'event_type': sly.EventType.PROJECT_CREATED, 'project_id': project_info.id})
    else:
        temp_str = "Project"
        if append_to_existing_project is True:
            temp_str = "Dataset"
        raise RuntimeError("{} wasn't created: 0 files with supported codecs ({}) and containers ({}). It is a limitation for Community Edition (CE)."
                           .format(temp_str, _SUPPORTED_CODECS, _SUPPORTED_CONTAINERS))
    pass


def main():
    add_videos_to_project()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('VIDEO_ONLY_IMPORT', main)