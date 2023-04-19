import json
import os
import random
import re
import string
import shutil
import tarfile
import requests
import subprocess
from pathlib import Path
import git
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm
from giturlparse import parse


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


def get_appKey(repo: git.Repo, sub_app_path: str):
    import hashlib

    remote_name = repo.active_branch.tracking_branch().remote_name
    print("\nDEBUG: remote_name =", remote_name, "\n")
    remote = repo.remote(remote_name)
    remote_url = get_remote_url(remote)
    print("\nDEBUG: remote_url =", remote_url, "\n")
    first_commit = next(repo.iter_commits("HEAD", reverse=True))
    print("\nDEBUG: first_commit =", first_commit.hexsha, "\n")
    key_string = remote_url + "_" + first_commit.hexsha
    print("\nDEBUG: key_string =", key_string, "\n")
    appKey = hashlib.md5(key_string.encode("utf-8")).hexdigest()
    if sub_app_path is not None:
        appKey += "_" + hashlib.md5(sub_app_path.encode("utf-8")).hexdigest()
    appKey += "_" + hashlib.md5(first_commit.hexsha[:7].encode("utf-8")).hexdigest()
    print("\nDEBUG: appKey =", appKey, "\n")
    
    return appKey


def get_instance_version(token, server):
    headers = {
        "x-api-key": token,
        "Content-Type": "application/json",
    }
    r = requests.post(
        f'{server.rstrip("/")}/public/api/v3/instance.version', headers=headers
    )
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
):
    f = open(archive_path, "rb")
    fields = {
        "appKey": appKey,
        "release": json.dumps(release),
        "config": json.dumps(config),
        "readme": readme,
        "modalTemplate": modal_template,
        "subAppPath": subapp_path,
        "archive": (
            "arhcive.tar.gz",
            f,
            "application/gzip",
        )
    }
    if slug:
        fields["slug"] = slug
    if user_id:
        fields["userId"] = str(user_id)
    e = MultipartEncoder(fields=fields)
    encoder_len = e.len
    with tqdm(
        total=encoder_len,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        m = MultipartEncoderMonitor(
            e, lambda monitor: bar.update(monitor.bytes_read - bar.n)
        )
        response = requests.post(
            f"{server_address.rstrip('/')}/public/api/v3/ecosystem.release",
            data=m,
            headers={"Content-Type": m.content_type, "x-api-key": api_token},
        )
    f.close()
    return response


def archivate_application(repo: git.Repo, config, slug):
    archive_folder = "".join(random.choice(string.ascii_letters) for _ in range(5))
    os.mkdir(archive_folder)
    entries = [entry for entry in repo.commit().tree.traverse()]
    entries = [entry for entry in entries if entry not in repo.ignored(entries)]
    if slug is None:
        app_folder_name = config["name"].lower()
    else:
        app_folder_name = slug.split("/")[1].lower()
    app_folder_name = re.sub("[ \/]", "-", app_folder_name)
    app_folder_name = re.sub("[\"'`,\[\]\(\)]", "", app_folder_name)
    with tarfile.open(archive_folder + "/archive.tar.gz", "w:gz") as tar:
        for entry in entries:
            if Path(entry.abspath).is_file():
                tar.add(entry.abspath, app_folder_name + "/" + entry.path)
    return archive_folder


def get_user(server_address, api_token):
    headers = {
        "x-api-key": api_token,
        "Content-Type": "application/json",
    }
    r = requests.post(
        f'{server_address.rstrip("/")}/public/api/v3/users.me', headers=headers
    )
    if r.status_code == 403:
        raise PermissionError()
    if r.status_code == 404 or r.status_code == 400:
        return None
    if r.status_code != 200:
        raise ConnectionError()
    return r.json()


def delete_directory(path):
    shutil.rmtree(path)


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
):
    archive_dir = archivate_application(repo, config, slug)
    release = {"name": release_name, "version": release_version}
    response = upload_archive(
        archive_dir + "/archive.tar.gz",
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
    )
    delete_directory(archive_dir)
    return response
