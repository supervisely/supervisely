import json
import os
from os.path import basename, join

import supervisely.io.env as env
import supervisely.io.env as sly_env
from supervisely import rand_str
from supervisely._utils import is_production
from supervisely.api.api import Api
from supervisely.app.fastapi import get_name_from_env
from supervisely.io.fs import (
    archive_directory,
    get_file_name_with_ext,
    is_archive,
    remove_dir,
    silent_remove,
)
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress
from supervisely.team_files import RECOMMENDED_EXPORT_PATH


def set_project(id: int):
    if is_production() is True:
        api = Api()
        task_id = sly_env.task_id()
        api.task.set_output_project(task_id, project_id=id)
    else:
        print(f"Output project: id={id}")


def set_directory(teamfiles_dir: str):
    """
    Sets a link to a teamfiles directory in workspace tasks interface
    """
    if is_production():
        api = Api()
        task_id = sly_env.task_id()

        if api.task.get_info_by_id(task_id) is None:
            raise KeyError(
                f"Task with ID={task_id} is either not exist or not found in your account"
            )

        team_id = api.task.get_info_by_id(task_id)["teamId"]

        if api.team.get_info_by_id(team_id) is None:
            raise KeyError(
                f"Team with ID={team_id} is either not exist or not found in your account"
            )

        files = api.file.list2(team_id, teamfiles_dir, recursive=True)

        # if directory is empty or not exists
        if len(files) == 0:
            # some data to create dummy .json file to get file id
            data = {"team_id": team_id, "task_id": task_id, "directory": teamfiles_dir}
            filename = f"{rand_str(10)}.json"

            src_path = os.path.join("/tmp/", filename)
            with open(src_path, "w") as f:
                json.dump(data, f)

            dst_path = os.path.join(teamfiles_dir, filename)
            file_id = api.file.upload(team_id, src_path, dst_path).id

            silent_remove(src_path)

        else:
            file_id = files[0].id

        api.task.set_output_directory(task_id, file_id, teamfiles_dir)

    else:
        print(f"Output directory: '{teamfiles_dir}'")


def set_download(local_path: str):
    """
    Receives a path to the local file or directory. If the path is a directory, it will be archived before uploading.
    After sets a link to a uploaded file in workspace tasks interface according to the file type.
    If the file is an archive, the set_output_archive method is called and "Download archive" text is displayed.
    If the file is not an archive, the set_output_file_download method is called and "Download file" text is displayed.

    :param local_path: path to the local file or directory, which will be uploaded to the teamfiles
    :type local_path: str
    :return: FileInfo object
    :rtype: FileInfo
    """
    if os.path.isdir(local_path):
        archive_path = f"{local_path}.tar"
        archive_directory(local_path, archive_path)
        remove_dir(local_path)
        local_path = archive_path

    if is_production():
        api = Api()
        task_id = sly_env.task_id()
        upload_progress = []

        team_id = env.team_id()

        def _print_progress(monitor, upload_progress):
            if len(upload_progress) == 0:
                upload_progress.append(
                    Progress(
                        message=f"Uploading '{basename(local_path)}'",
                        total_cnt=monitor.len,
                        ext_logger=logger,
                        is_size=True,
                    )
                )
            upload_progress[0].set_current_value(monitor.bytes_read)

        remote_path = join(
            RECOMMENDED_EXPORT_PATH,
            get_name_from_env(),
            str(task_id),
            f"{get_file_name_with_ext(local_path)}",
        )
        file_info = api.file.upload(
            team_id=team_id,
            src=local_path,
            dst=remote_path,
            progress_cb=lambda m: _print_progress(m, upload_progress),
        )

        if is_archive(local_path):
            api.task.set_output_archive(task_id, file_info.id, file_info.name)
        else:
            api.task.set_output_file_download(task_id, file_info.id, file_info.name)

        logger.info(f"Remote file: id={file_info.id}, name={file_info.name}")
        silent_remove(local_path)
        return file_info
    else:
        logger.info(f"Output file: '{local_path}'")
