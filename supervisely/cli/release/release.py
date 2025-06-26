import datetime
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import tarfile
from pathlib import Path

import git
import requests
from giturlparse import parse
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm

from supervisely.io.fs import dir_exists, list_files_recursively, remove_dir


class cd:
    def __init__(self, new_path=None, add_to_path=False):
        self.new_path = new_path
        self.add_to_path = add_to_path
        self._should_remove_from_path = False

    def __enter__(self):
        self.old_path = os.getcwd()
        if self.new_path is not None:
            os.chdir(self.new_path)
        if self.add_to_path and not self.new_path in sys.path:
            sys.path.insert(0, self.new_path)
            self._should_remove_from_path = True
        return self

    def __exit__(self, etype, value, traceback):
        os.chdir(self.old_path)
        if self._should_remove_from_path:
            sys.path.remove(self.new_path)


def slug_is_valid(slug):
    splitted = slug.split("/")
    return len(splitted) == 2 and len(splitted[0]) > 0 and len(splitted[1]) > 0


def get_module_root(path: str):
    if path is None:
        return Path(os.getcwd())
    return Path(path).absolute().resolve()


def get_module_path(module_root, sub_app):
    if sub_app is None:
        return module_root
    return module_root.joinpath(sub_app)


def get_remote_url(remote: git.Remote):
    p = parse(remote.url)
    url = p.url2https.replace("https://", "").replace(".git", "").lower()
    return url


def get_semver(string):
    return re.match("\d+\.\d+\.\d+", string)


def find_tag_in_repo(tag_name, repo: git.Repo):
    for tag in repo.tags:
        if tag.name == tag_name:
            return tag
    return None


def push_tag(tag_name, repo: git.Repo):
    remote_name = repo.active_branch.tracking_branch().remote_name
    command = f"git push --porcelain -- {remote_name} {tag_name}"
    try:
        subprocess.check_call(command, shell=True, cwd=repo.working_dir)
    except subprocess.CalledProcessError:
        raise


def delete_tag(tag_name, repo: git.Repo):
    tag = find_tag_in_repo(tag_name, repo)
    repo.delete_tag(tag)
    remote_name = remote_name = repo.active_branch.tracking_branch().remote_name
    command = f"git push --porcelain -- {remote_name} :refs/tags/{tag_name}"
    try:
        subprocess.check_call(command, shell=True, cwd=repo.working_dir)
    except subprocess.CalledProcessError:
        raise


def get_appKey(repo, sub_app_path, repo_url):
    import hashlib

    p = parse(repo_url)
    repo_url = p.url2https.replace("https://", "").replace(".git", "").lower()

    first_commit = next(repo.iter_commits("HEAD", reverse=True))
    key_string = repo_url + "_" + first_commit.hexsha
    appKey = hashlib.md5(key_string.encode("utf-8")).hexdigest()
    if sub_app_path is not None:
        appKey += "_" + hashlib.md5(sub_app_path.encode("utf-8")).hexdigest()
    appKey += "_" + hashlib.md5(first_commit.hexsha[:7].encode("utf-8")).hexdigest()

    return appKey


def get_instance_version(token, server):
    headers = {
        "x-api-key": token,
        "Content-Type": "application/json",
    }
    r = requests.post(f'{server.rstrip("/")}/public/api/v3/instance.version', headers=headers)
    if r.status_code == 403:
        raise PermissionError()
    if r.status_code == 404:
        raise NotImplementedError()
    if r.status_code != 200:
        raise ConnectionError()
    return r.json()


def get_app_from_instance(appKey: str, token, server):
    headers = {
        "x-api-key": token,
        "Content-Type": "application/json",
    }
    data = json.dumps(
        {
            "appKey": appKey,
        }
    )
    r = requests.post(
        f'{server.rstrip("/")}/public/api/v3/ecosystem.info', headers=headers, data=data
    )
    if r.status_code == 403:
        raise PermissionError()
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise ConnectionError()
    return r.json()


def upload_archive(
    archive_path,
    server_address,
    api_token,
    appKey,
    release,
    config,
    readme,
    modal_template,
    slug,
    user_id,
    subapp_path,
    share_app,
    files,
):
    f = open(archive_path, "rb")
    archive_name = os.path.basename(archive_path)
    fields = {
        "appKey": appKey,
        "subAppPath": subapp_path,
        "release": json.dumps(release),
        "config": json.dumps(config),
        "readme": readme,
        "modalTemplate": modal_template,
        "archive": (
            archive_name,
            f,
            "application/gzip" if archive_name.endswith(".tar.gz") else "application/x-tar",
        ),
    }
    if slug:
        fields["slug"] = slug
    if user_id:
        fields["userId"] = str(user_id)
    if share_app:
        fields["isShared"] = "true"
    if files:
        files_contents = {}
        fields["files"] = files_contents
        for file_name, file_path in files.items():
            files_contents[file_name] = Path(file_path).read_text(encoding="utf-8")
        fields["files"] = json.dumps(files_contents)

    e = MultipartEncoder(fields=fields)
    encoder_len = e.len
    with tqdm(
        total=encoder_len,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        m = MultipartEncoderMonitor(e, lambda monitor: bar.update(monitor.bytes_read - bar.n))
        response = requests.post(
            f"{server_address.rstrip('/')}/public/api/v3/ecosystem.release",
            data=m,
            headers={"Content-Type": m.content_type, "x-api-key": api_token},
        )
    f.close()
    return response


def archive_application(repo: git.Repo, config, slug):
    archive_folder = "".join(random.choice(string.ascii_letters) for _ in range(5))
    os.mkdir(archive_folder)
    file_paths = [
        Path(line.decode("utf-8")).absolute()
        for line in subprocess.check_output(
            "git ls-files --recurse-submodules", shell=True
        ).splitlines()
    ]
    if slug is None:
        app_folder_name = config["name"].lower()
    else:
        app_folder_name = slug.split("/")[1].lower()
    app_folder_name = re.sub("[ \/]", "-", app_folder_name)
    app_folder_name = re.sub("[\"'`,\[\]\(\)]", "", app_folder_name)
    working_dir_path = Path(repo.working_dir).absolute()
    should_remove_dir = None
    if config.get("type", "app") == "client_side_app":
        gui_folder_path = config["gui_folder_path"]
        gui_folder_path = working_dir_path / gui_folder_path
        if not dir_exists(gui_folder_path):
            should_remove_dir = gui_folder_path
            # if gui folder is empty, need to render it
            with cd(str(working_dir_path), add_to_path=True):
                exec(open("sly_sdk/render.py", "r").read(), {"__name__": "__main__"})
                file_paths.extend(
                    [Path(p).absolute() for p in list_files_recursively(str(gui_folder_path))]
                )
        archive_path = archive_folder + "/archive.tar"
        write_mode = "w"
    else:
        archive_path = archive_folder + "/archive.tar.gz"
        write_mode = "w:gz"
    with tarfile.open(archive_path, write_mode) as tar:
        for path in file_paths:
            if path.is_file():
                tar.add(
                    path.absolute(),
                    Path(app_folder_name).joinpath(path.relative_to(working_dir_path)),
                )
    if should_remove_dir is not None:
        # remove gui folder if it was rendered
        remove_dir(should_remove_dir)
    return archive_path


def get_user(server_address, api_token):
    headers = {
        "x-api-key": api_token,
        "Content-Type": "application/json",
    }
    r = requests.post(f'{server_address.rstrip("/")}/public/api/v3/users.me', headers=headers)
    if r.status_code == 403:
        raise PermissionError()
    if r.status_code == 404 or r.status_code == 400:
        return None
    if r.status_code != 200:
        raise ConnectionError()
    return r.json()


def delete_directory(path):
    shutil.rmtree(path)


def get_created_at(repo: git.Repo, tag_name):
    if tag_name is None:
        return None
    for tag in repo.tags:
        if tag.name == tag_name:
            if tag.tag is None:
                timestamp = tag.commit.committed_date
            else:
                timestamp = tag.tag.tagged_date
            return datetime.datetime.utcfromtimestamp(timestamp).isoformat()
    return None


def release(
    server_address,
    api_token,
    appKey,
    repo: git.Repo,
    config,
    readme,
    release_name,
    release_version,
    modal_template="",
    slug=None,
    user_id=None,
    subapp_path="",
    created_at=None,
    share_app=False,
    files=None,
):
    if created_at is None:
        created_at = get_created_at(repo, release_version)
    archive_path = archive_application(repo, config, slug)
    release = {
        "name": release_name,
        "version": release_version,
    }
    if created_at is not None:
        release["createdAt"] = created_at
    try:
        response = upload_archive(
            archive_path,
            server_address,
            api_token,
            appKey,
            release,
            config,
            readme,
            modal_template,
            slug,
            user_id,
            subapp_path,
            share_app,
            files,
        )
    finally:
        delete_directory(os.path.dirname(archive_path))
    return response
