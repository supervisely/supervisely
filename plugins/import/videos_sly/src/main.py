# coding: utf-8
import os
from supervisely_lib.io.json import load_json_file
from supervisely_lib import TaskPaths
import supervisely_lib as sly
from supervisely_lib.video.video import _check_video_requires_processing, warn_video_requires_processing, \
                                        get_video_streams

DEFAULT_DATASET_NAME = 'ds0'
root_ds_name = DEFAULT_DATASET_NAME


def add_videos_to_project():
    task_config = load_json_file(TaskPaths.TASK_CONFIG_PATH)

    task_id = task_config['task_id']
    append_to_existing_project = task_config['append_to_existing_project']
    if append_to_existing_project is True:
        raise RuntimeError("Appending to existing project is not supported by this version")

    server_address = task_config['server_address']
    token = task_config['api_token']

    instance_type = task_config.get("instance_type", sly.ENTERPRISE)

    api = sly.Api(server_address, token)

    task_info = api.task.get_info_by_id(task_id)
    api.add_additional_field('taskId', task_id)
    api.add_header('x-task-id', str(task_id))

    workspace_id = task_info["workspaceId"]
    project_name = task_config['project_name']

    project_dir = os.path.join(sly.TaskPaths.DATA_DIR, project_name)
    path_info_map = {}

    files_list = api.task.get_import_files_list(task_id)
    for file_info in files_list:
        original_path = file_info["filename"]
        ext = sly.fs.get_file_ext(original_path)
        save_path = project_dir + original_path
        path_info_map[save_path] = file_info
        if ext == '.json':
            api.task.download_import_file(task_id, original_path, save_path)
        else:
            sly.video.validate_ext(ext)
            #@TODO: validate streams for community
            sly.fs.touch(save_path)

    # files structure without original video files is done
    # validate project structure
    project_fs = sly.VideoProject.read_single(sly.TaskPaths.DATA_DIR)

    project = api.project.create(workspace_id, project_name, type=sly.ProjectType.VIDEOS, change_name_if_conflict=True)
    api.project.update_meta(project.id, project_fs.meta.to_json())
    sly.logger.info("Project {!r} [id={!r}] has been created".format(project.name, project.id))

    for dataset_fs in project_fs:
        dataset = api.dataset.get_info_by_name(project.id, dataset_fs.name)
        if dataset is None:
            dataset = api.dataset.create(project.id, dataset_fs.name)
            sly.logger.info("dataset {!r} [id={!r}] has been created".format(dataset.name, dataset.id))

        for item_name in dataset_fs:
            item_path, ann_path = dataset_fs.get_item_paths(item_name)

            file_info = path_info_map[item_path]
            video_streams = get_video_streams(file_info["meta"]["streams"])
            if len(video_streams) != 1:
                sly.logger.warn(("Video {!r} has {} video streams. Import Videos in Supervisely format supports only"
                                 "videos with a single video stream. And annotation file has to be provided"
                                 "for each stream. Item will be skipped.")
                                .format(item_path, len(video_streams)))
                continue

            if instance_type is sly.COMMUNITY:
                if _check_video_requires_processing(file_info, video_streams[0]) is True:
                    warn_video_requires_processing(item_path)
                    continue

            item_hash = file_info["hash"]
            item_meta = {}

            video = api.video.upload_hash(dataset.id, item_name, item_hash, item_meta)

            #validate_item_annotation
            ann_json = sly.io.json.load_json_file(ann_path)
            ann = sly.VideoAnnotation.from_json(ann_json, project_fs.meta)

            # ignore existing key_id_map because the new objects will be created
            api.video.annotation.append(video.id, ann)

    sly.logger.info('PROJECT_CREATED', extra={'event_type': sly.EventType.PROJECT_CREATED, 'project_id': project.id})
    pass


def main():
    add_videos_to_project()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('VIDEO_SLY_IMPORT', main)
