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
from rich.console import Console
from tqdm import tqdm

from supervisely.io.fs import dir_exists, list_files_recursively, remove_dir

console = Console()


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
    """
    Download an archive into memory, for post-upload verification. Returns
    (data, error_message) - never raises: a timeout or a connection dropping
    mid-stream is reported as an error string, not an exception, so a transient
    network blip during *verification* can never discard an upload that already
    succeeded.
    """
    payload = {
        "moduleId": ecosystem_item_id,
        "version": version,
        "isArchive": True,
    }
    try:
        resp = requests.post(
            f"{server_address.rstrip('/')}/public/api/v3/ecosystem.file.download",
            json=payload,
            headers={"x-api-key": api_token},
            stream=True,
            timeout=60,
        )
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code} while reading back the uploaded archive"
        buf = io.BytesIO()
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            buf.write(chunk)
        return buf.getvalue(), None
    except requests.exceptions.RequestException as e:
        return None, f"{type(e).__name__} while reading back the uploaded archive: {e}"


def _validate_archive_bytes(data):
    """
    Structural validation of an in-memory archive. Uses tarfile's auto-detecting
    "r:*" mode and actually iterates every member (not just opening the stream),
    so this covers both the gzip-compressed (.tar.gz) and plain (.tar, used for
    client_side_app releases) cases, and catches the case where the outer gzip
    envelope is technically well-formed but the tar content inside it is
    truncated - a bare gzip-only check would miss that.

    Caveat: member iteration alone does not always catch a stream truncated
    exactly on a member-header boundary (verified: a 2-member plain tar cut
    right before the 2nd header validates clean as a 1-member tar). The caller
    is expected to also compare the read-back size against what was actually
    uploaded - do not rely on this function alone to catch that shape.
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
    expected_size,
    max_read_attempts=3,
    initial_backoff_sec=3,
):
    """
    Read an uploaded archive back from the server and confirm it is complete and
    structurally valid. Neither the upload endpoint nor the download endpoint
    validate transfer completeness on their own - a connection that drops
    mid-transfer can still result in a "successful" HTTP response with a
    truncated file stored server-side.

    Returns (status, error_message), where status is one of:
      - "valid": read-back size matches what was uploaded AND the structure
        validates. The only outcome that is fully confirmed-good.
      - "corrupt": read-back size matches, but the structure is broken. This is
        unambiguous - the read is not racing anything, so retrying the read
        further cannot help; this is real, actionable corruption.
      - "inconclusive": everything else - network/permission errors, or a size
        mismatch (with or without a structural error), even after retries. This
        is deliberately NOT escalated to "corrupt": storage/indexing on the
        server can be asynchronous (a read immediately after the 200 response
        can race the server and 404 or serve a previous version's bytes), and
        the server may also legitimately transform the archive on store (e.g.
        re-compress it), so a size difference alone is not proof of corruption.
        Callers must not treat "inconclusive" as a hard failure.

    Retries the read itself (not the upload) with exponential backoff.
    """
    backoff = initial_backoff_sec
    last_err = "no read-back attempts were made"
    ecosystem_item_id = None
    for attempt in range(1, max_read_attempts + 1):
        if ecosystem_item_id is None:
            try:
                app_info = get_app_from_instance(appKey, api_token, server_address)
            except (PermissionError, ConnectionError, NotImplementedError) as e:
                app_info = None
                last_err = f"could not resolve ecosystem item id: {type(e).__name__}"
            if app_info is not None and "id" in app_info:
                ecosystem_item_id = app_info["id"]
            elif app_info is None:
                last_err = "could not resolve ecosystem item id for appKey"

        if ecosystem_item_id is not None:
            data, err = _download_archive_bytes(
                server_address, api_token, ecosystem_item_id, version
            )
            if err is not None:
                last_err = err
            else:
                size_matches = len(data) == expected_size
                is_valid, struct_err = _validate_archive_bytes(data)
                if size_matches and is_valid:
                    return "valid", None
                if size_matches and not is_valid:
                    # Exact size match rules out a propagation race or a
                    # server-side transform changing the byte count - this is
                    # a confirmed, unambiguous corruption.
                    return "corrupt", struct_err
                last_err = (
                    f"read-back size {len(data)} != uploaded size {expected_size}"
                    + ("" if is_valid else f"; also structurally invalid ({struct_err})")
                )

        if attempt < max_read_attempts:
            time.sleep(backoff)
            backoff *= 2
    return "inconclusive", f"gave up after {max_read_attempts} read-back attempts: {last_err}"


def _is_duplicate_version_response(response):
    """
    True only for the precise "version ... already exists" server response shape
    (matching what run.py's own tag-deletion guard checks for) - not a loose
    substring match, which could also match an unrelated "file already exists"
    or similar message elsewhere in the payload.
    """
    try:
        message = response.json()["details"]["message"]
    except Exception:
        return False
    message = message.strip().lower()
    return message.startswith("version") and message.endswith("already exists")


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
    archive_name = os.path.basename(archive_path)
    version = release.get("version")
    expected_size = os.path.getsize(archive_path)

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
            m = MultipartEncoderMonitor(e, lambda monitor: bar.update(monitor.bytes_read - bar.n))
            response = requests.post(
                f"{server_address.rstrip('/')}/public/api/v3/ecosystem.release",
                data=m,
                headers={"Content-Type": m.content_type, "x-api-key": api_token},
            )

    if not response.ok:
        # Deliberately NOT retried, including for a duplicate-version rejection:
        # this is the *only* upload attempt this call makes, so a "version
        # already exists" response here can only mean a genuinely pre-existing
        # release (e.g. re-releasing an already-published version) - never
        # something this same call caused. Returning the response as-is
        # (unchanged from the pre-verification behavior) lets run.py's own
        # message.startswith("version")/endswith("already exists") check
        # correctly recognize that shape and skip deleting the just-created
        # git tag. Retrying here would only re-POST the identical version and
        # get rejected identically - it cannot help even for real corruption.
        return response

    if version is None:
        # Nothing to read back and compare against (e.g. archive_only_config flows).
        return response

    status, err = _read_back_and_verify(server_address, api_token, appKey, version, expected_size)
    if status == "corrupt":
        raise RuntimeError(
            f"Uploaded archive for appKey={appKey!r} version={version!r} was confirmed "
            f"corrupted server-side: {err}. The read-back byte size matched exactly what "
            "was uploaded, so this is not a propagation race or a transient network issue - "
            "the ecosystem.release/ecosystem.file.download backend needs investigation. "
            "Re-uploading under the same version will be rejected as a duplicate, so this "
            "cannot be fixed by retrying; a genuinely new version (or server-side "
            "intervention) is required."
        )
    if status == "inconclusive":
        console.print(
            f"[orange1][Warning][/] Could not conclusively verify the archive uploaded for "
            f"version {version!r} ({err}). Proceeding without blocking the release - a size "
            "or availability mismatch on read-back is not proof of corruption (e.g. "
            "asynchronous server-side storage/indexing, or the server legitimately "
            "re-packing the archive on store)."
        )
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
