# coding: utf-8
import os
from supervisely_lib.io.json import load_json_file
from supervisely_lib import TaskPaths
import supervisely_lib as sly
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.video_annotation.key_id_map import KeyIdMap

DEFAULT_DATASET_NAME = 'ds0'
root_ds_name = DEFAULT_DATASET_NAME


def add_pointclouds_to_project():
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
            if sly.PointcloudDataset.related_images_dir_name in save_path:
                sly.image.validate_ext(save_path)
            else:
                sly.pointcloud.validate_ext(ext)
            sly.fs.touch(save_path)

    # files structure without original video files is done
    # validate project structure

    project_fs = sly.PointcloudProject.read_single(sly.TaskPaths.DATA_DIR)

    project = api.project.create(workspace_id, project_name, type=sly.ProjectType.POINT_CLOUDS, change_name_if_conflict=True)
    api.project.update_meta(project.id, project_fs.meta.to_json())
    sly.logger.info("Project {!r} [id={!r}] has been created".format(project.name, project.id))

    uploaded_objects = KeyIdMap()
    for dataset_fs in project_fs:
        dataset = api.dataset.get_info_by_name(project.id, dataset_fs.name)
        if dataset is None:
            dataset = api.dataset.create(project.id, dataset_fs.name)
            sly.logger.info("dataset {!r} [id={!r}] has been created".format(dataset.name, dataset.id))

        for item_name in dataset_fs:
            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)

            file_info = path_info_map[item_path]
            item_hash = file_info[ApiField.HASH]
            item_meta = {}

            pointcloud = api.pointcloud.upload_hash(dataset.id, item_name, item_hash, item_meta)

            #validate_item_annotation
            ann_json = sly.io.json.load_json_file(ann_path)
            ann = sly.PointcloudAnnotation.from_json(ann_json, project_fs.meta)

            # ignore existing key_id_map because the new objects will be created
            api.pointcloud.annotation.append(pointcloud.id, ann, uploaded_objects)

            #upload related_images if exist
            related_items = dataset_fs.get_related_images(item_name)
            if len(related_items) != 0:
                rimg_infos = []
                for img_path, meta_json in related_items:
                    rimg_infos.append({ApiField.ENTITY_ID: pointcloud.id,
                                       ApiField.NAME: meta_json[ApiField.NAME],
                                       ApiField.HASH: path_info_map[img_path][ApiField.HASH],
                                       ApiField.META: meta_json[ApiField.META],})
                api.pointcloud.add_related_images(rimg_infos)

    sly.logger.info('PROJECT_CREATED', extra={'event_type': sly.EventType.PROJECT_CREATED, 'project_id': project.id})
    pass


def main():
    add_pointclouds_to_project()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('POINTCLOUD_SLY_IMPORT', main)
