import os
import shutil
import tempfile
import traceback

import supervisely as sly

import functools
import pathlib
from collections import namedtuple
from distutils.dir_util import copy_tree

from fastapi import FastAPI
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles


def get_static_paths_by_mounted_object(mount) -> list:
    StaticPath = namedtuple('StaticPath', ['local_path', 'url_path'])
    static_paths = []

    if hasattr(mount, 'routes'):
        for current_route in mount.routes:
            if type(current_route) == Mount and type(current_route.app) == FastAPI:
                all_children_paths = get_static_paths_by_mounted_object(current_route)
                for index, current_path in enumerate(all_children_paths):
                    current_url_path = pathlib.Path(str(current_route.path).lstrip('/'), str(current_path.url_path).lstrip('/'))
                    all_children_paths[index] = StaticPath(local_path=current_path.local_path, url_path=current_url_path)
                static_paths.extend(all_children_paths)
            elif type(current_route) == Mount and type(current_route.app) == StaticFiles:
                static_paths.append(StaticPath(local_path=pathlib.Path(current_route.app.directory),
                                               url_path=pathlib.Path(str(current_route.path).lstrip('/'))))

    return static_paths


def dump_statics_to_dir(static_dir_path: pathlib.Path, static_paths: list):
    for current_path in static_paths:
        current_local_path: pathlib.Path = current_path.local_path
        current_url_path: pathlib.Path = static_dir_path / current_path.url_path

        if current_local_path.is_dir():
            current_url_path.mkdir(parents=True, exist_ok=True)
            copy_tree(current_local_path.as_posix(), current_url_path.as_posix(), preserve_symlinks=True)


def dump_html_to_dir(static_dir_path, template):
    pathlib.Path(static_dir_path / template.template.name).write_bytes(template.body)


def available_after_shutdown(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            template_response, app = f(*args, **kwargs)

            app_template_path = pathlib.Path(tempfile.mkdtemp())
            app_static_paths = get_static_paths_by_mounted_object(mount=app)
            dump_statics_to_dir(static_dir_path=app_template_path, static_paths=app_static_paths)
            dump_html_to_dir(static_dir_path=app_template_path, template=template_response)

            # upload to supervisely here
            # remote_dir = pathlib.Path(os.getenv('APP_NAME', 'sly_app'), os.getenv('TASK_ID', '0000'), 'app-template')

            shutil.rmtree(app_template_path.as_posix())
            sly.logger.info(f'App files stored in {app_template_path} for offline usage')

            return template_response
        except Exception as ex:
            traceback.print_exc()
            sly.logger.warning(f'Cannot dump files for offline usage, reason: {ex}')

    return wrapper
