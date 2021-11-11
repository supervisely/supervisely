# coding: utf-8

import os
from supervisely_lib.io.json import load_json_file
from supervisely_lib import TaskPaths
import supervisely_lib as sly
from supervisely_lib.video.import_utils import get_dataset_name
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.pointcloud.pointcloud import is_valid_ext, ALLOWED_POINTCLOUD_EXTENSIONS
from supervisely_lib.imaging import image

DEFAULT_DATASET_NAME = 'ds0'
root_ds_name = DEFAULT_DATASET_NAME


def add_pointclouds_to_project():
    task_config = load_json_file(TaskPaths.TASK_CONFIG_PATH)

    task_id = task_config['task_id']
    append_to_existing_project = task_config['append_to_existing_project']

    server_address = task_config['server_address']
    token = task_config['api_token']

    #instance_type = task_config.get("instance_type", sly.ENTERPRISE)

    api = sly.Api(server_address, token)

    task_info = api.task.get_info_by_id(task_id)
    api.add_additional_field('taskId', task_id)
    api.add_header('x-task-id', str(task_id))

    workspace_id = task_info["workspaceId"]
    project_name = task_config.get('project_name')
    if project_name is None:
        project_name = task_config["res_names"]["project"]

    project_info = None
    if append_to_existing_project is True:
        project_info = api.project.get_info_by_name(workspace_id, project_name, expected_type=sly.ProjectType.POINT_CLOUDS, raise_error=True)

    files_list = api.task.get_import_files_list(task_id)

    #find related images
    related_items_info = {} #item_dir->item_name_processed->[img or json info]
    related_items = {}
    for file_info in files_list:
        original_path = file_info["filename"]
        if 'related_images' in original_path:
            related_items[original_path] = file_info
            item_dir = original_path.split('/related_images')[0]
            item_name_processed = os.path.basename(os.path.dirname(original_path))

            if item_dir not in related_items_info:
                related_items_info[item_dir] = {}
            if item_name_processed not in related_items_info[item_dir]:
                related_items_info[item_dir][item_name_processed] = []
            related_items_info[item_dir][item_name_processed].append(file_info)

    added_items = []
    for file_info in files_list:
        ds_info = None
        original_path = file_info["filename"]
        if original_path in related_items:
            continue

        try:
            file_name = sly.fs.get_file_name_with_ext(original_path)
            ext = sly.fs.get_file_ext(original_path)

            hash = file_info["hash"]

            if is_valid_ext(ext):
                if project_info is None:
                    project_info = api.project.create(workspace_id, project_name, type=sly.ProjectType.POINT_CLOUDS, change_name_if_conflict=True)
                if ds_info is None:
                    ds_name = get_dataset_name(original_path)
                    ds_info = api.dataset.get_or_create(project_info.id, ds_name)
                item_name = api.pointcloud.get_free_name(ds_info.id, file_name)
                item_info = api.pointcloud.upload_hash(ds_info.id, item_name, hash)
                added_items.append((ds_info, item_info, original_path))
        except Exception as e:
            sly.logger.warning("File skipped {!r}: error occurred during processing {!r}".format(original_path, str(e)))

    # add related images for all added items
    for ds_info, item_info, import_path in added_items:
        item_dir = os.path.dirname(import_path)
        item_import_name = sly.fs.get_file_name_with_ext(import_path)
        item_context_dir = item_import_name.replace(".", "_")

        if item_dir not in related_items_info:
            continue
        if item_context_dir not in related_items_info[item_dir]:
            continue

        files = related_items_info[item_dir][item_context_dir]
        temp_dir = os.path.join(sly.TaskPaths.DATA_DIR, item_context_dir)
        sly.fs.mkdir(temp_dir)
        context_img_to_hash = {}
        for file_import_info in files:
            original_path = file_import_info["filename"]
            save_name = sly.fs.get_file_name_with_ext(original_path)
            api.task.download_import_file(task_id, original_path, os.path.join(temp_dir, save_name))
            context_img_to_hash[os.path.join(temp_dir, save_name)] = file_import_info

        related_items = []
        files = sly.fs.list_files(temp_dir, sly.image.SUPPORTED_IMG_EXTS)
        for file in files:
            img_meta_path = os.path.join(temp_dir, sly.fs.get_file_name_with_ext(file) + ".json")
            img_meta = {}
            if sly.fs.file_exists(img_meta_path):
                img_meta = load_json_file(img_meta_path)
                if not image.has_valid_ext(img_meta[ApiField.NAME]):
                    raise RuntimeError('Wrong format: name field contains path with unsupported extension')
                if img_meta[ApiField.NAME] != sly.fs.get_file_name_with_ext(file):
                    raise RuntimeError('Wrong format: name field contains wrong image path')
            related_items.append((file, img_meta, context_img_to_hash[file]['hash']))

        if len(related_items) != 0:
            rimg_infos = []
            for img_path, meta_json, hash in related_items:
                rimg_infos.append({ApiField.ENTITY_ID: item_info.id,
                                   ApiField.NAME: meta_json.get(ApiField.NAME, sly.fs.get_file_name_with_ext(img_path)),
                                   ApiField.HASH: hash,
                                   ApiField.META: meta_json.get(ApiField.META, {}) })
            api.pointcloud.add_related_images(rimg_infos)

        sly.fs.remove_dir(temp_dir)

        pass

    if project_info is not None:
        sly.logger.info('PROJECT_CREATED', extra={'event_type': sly.EventType.PROJECT_CREATED, 'project_id': project_info.id})
    else:
        temp_str = "Project"
        if append_to_existing_project is True:
            temp_str = "Dataset"
        raise RuntimeError("{} wasn't created: 0 files with supported formats were found. Supported formats: {!r}"
                           .format(temp_str, ALLOWED_POINTCLOUD_EXTENSIONS))
    pass


def main():
    add_pointclouds_to_project()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('POINTCLOUD_RAW_IMPORT', main)
