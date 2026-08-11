import datetime
import io
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import git
import requests
from giturlparse import parse
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm

from supervisely.io.fs import dir_exists, list_files_recursively, remove_dir


class cd:
    """Context manager that changes current working directory and optionally adds it to sys.path."""

    def __init__(self, new_path=None, add_to_path=False):
        """
        :param new_path: Directory to chdir into. If None, path is unchanged.
        :type new_path: str, optional
        :param add_to_path: If True, prepend new_path to sys.path on enter.
        :type add_to_path: bool
        """
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
    return re.match(r"\d+\.\d+\.\d+", string)


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


def _download_archive_bytes(server_address, api_token, ecosystem_item_id, version):
    """Download an archive into memory, for post-upload verification."""
    payload = {
        "moduleId": ecosystem_item_id,
        "version": version,
        "isArchive": True,
    }
    resp = requests.post(
        f"{server_address.rstrip('/')}/public/api/v3/ecosystem.file.download",
        json=payload,
        headers={"x-api-key": api_token},
        stream=True,
    )
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code} while reading back the uploaded archive"
    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        buf.write(chunk)
    return buf.getvalue(), None


def _validate_archive_bytes(data, archive_name):
    """
    Structural validation of an in-memory archive. Uses tarfile's auto-detecting
    "r:*" mode and actually iterates every member (not just opening the stream),
    so this covers both the gzip-compressed (.tar.gz) and plain (.tar, used for
    client_side_app releases) cases, and catches the case where the outer gzip
    envelope is technically well-formed but the tar content inside it is
    truncated - a bare gzip-only check would miss that.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            for _ in tar:
                pass
    except (EOFError, OSError, tarfile.TarError) as e:
        return False, f"downloaded archive is not a valid/complete tar stream: {e}"
    return True, None


def _read_back_and_verify(
    server_address,
    api_token,
    appKey,
    version,
    archive_name,
    expected_size,
    max_read_attempts=4,
    initial_backoff_sec=2,
):
    """
    Read the just-uploaded archive back from the server and confirm it is complete
    and structurally valid. Neither the upload endpoint nor the download endpoint
    validate transfer completeness on their own - a connection that drops
    mid-transfer can still result in a "successful" HTTP response with a truncated
    file stored server-side.

    Storage/indexing on the server can also be asynchronous: reading back
    *immediately* after the 200 response can race the server and either 404 or
    serve a previous version's bytes, which would look identical to real
    corruption unless we check for it. So this retries the read itself (not the
    upload) with exponential backoff, and requires the returned size to match the
    just-uploaded local archive's size before doing any deeper structural check -
    a size mismatch is treated as "not propagated yet" and retried, not as
    corruption. Only a size-matched-but-structurally-broken result is treated as
    a real, immediate failure (no point retrying the read further at that point).

    Returns (is_valid, error_message).
    """
    backoff = initial_backoff_sec
    last_err = "no read-back attempts were made"
    for attempt in range(1, max_read_attempts + 1):
        app_info = get_app_from_instance(appKey, api_token, server_address)
        if app_info is None or "id" not in app_info:
            last_err = "could not resolve ecosystem item id for appKey"
        else:
            data, err = _download_archive_bytes(
                server_address, api_token, app_info["id"], version
            )
            if err is not None:
                last_err = err
            elif len(data) != expected_size:
                last_err = (
                    f"read-back size {len(data)} does not match the uploaded size "
                    f"{expected_size} (server storage/indexing may still be propagating)"
                )
            else:
                return _validate_archive_bytes(data, archive_name)
        if attempt < max_read_attempts:
            time.sleep(backoff)
            backoff *= 2
    return False, f"gave up after {max_read_attempts} read-back attempts: {last_err}"


def _is_duplicate_version_response(response):
    """True if the server rejected the upload because this version was already released."""
    try:
        message = json.dumps(response.json())
    except Exception:
        message = response.text or ""
    message = message.lower()
    return "already exists" in message or "already released" in message


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
    max_attempts=2,
):
    archive_name = os.path.basename(archive_path)
    version = release.get("version")
    expected_size = os.path.getsize(archive_path)
    response = None
    last_ok_response = None

    for attempt in range(1, max_attempts + 1):
        with open(archive_path, "rb") as f:
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
                    "application/gzip"
                    if archive_name.endswith(".tar.gz")
                    else "application/x-tar",
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
                m = MultipartEncoderMonitor(
                    e, lambda monitor: bar.update(monitor.bytes_read - bar.n)
                )
                response = requests.post(
                    f"{server_address.rstrip('/')}/public/api/v3/ecosystem.release",
                    data=m,
                    headers={"Content-Type": m.content_type, "x-api-key": api_token},
                )

        if response.ok:
            last_ok_response = response
        else:
            if version is None or not _is_duplicate_version_response(response):
                # A genuine, unrelated rejection - let the existing caller-level
                # retry/error handling (do_release_with_retry) deal with it as before.
                return response
            # The server says this version is already released - almost certainly
            # because *this function's own previous attempt* already stored it
            # (retrying with the same version would otherwise never legitimately
            # hit this). Re-uploading again under the same version cannot help,
            # so check what is actually stored instead of treating this as fatal.
            print(
                f"[upload_archive] Attempt {attempt}/{max_attempts}: server reports version "
                f"{version!r} already exists; checking whether it was actually stored intact "
                "before treating this as a failure...",
                file=sys.stderr,
            )

        if version is None:
            # Nothing to read back and compare against (e.g. archive_only_config flows).
            return response

        is_valid, err = _read_back_and_verify(
            server_address, api_token, appKey, version, archive_name, expected_size
        )
        if is_valid:
            # response itself may be the "already exists" rejection from this attempt
            # (see above) - report the earlier successful upload's response in that
            # case, not the rejection, since the verified-good content is what matters.
            return last_ok_response if last_ok_response is not None else response

        print(
            f"[upload_archive] Attempt {attempt}/{max_attempts}: stored archive for version "
            f"{version!r} failed verification: {err}. "
            + ("Retrying with a fresh upload..." if attempt < max_attempts else "Giving up."),
            file=sys.stderr,
        )

    raise RuntimeError(
        f"Uploaded archive for appKey={appKey!r} version={version!r} repeatedly failed "
        f"post-upload verification after {max_attempts} attempts - the server is storing a "
        "truncated/corrupted archive despite returning a successful HTTP response. This is "
        "not a client-side problem to retry away; the ecosystem.release/ecosystem.file.download "
        "backend needs investigation."
    )


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
    app_folder_name = re.sub(r"[ /]", "-", app_folder_name)
    app_folder_name = re.sub(r"[\"'`,\[\]()]", "", app_folder_name)
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
