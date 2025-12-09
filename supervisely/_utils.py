# coding: utf-8

import asyncio
import base64
import copy
import hashlib
import inspect
import json
import os
import random
import re
import string
import time
import urllib
from datetime import datetime
from functools import wraps
from tempfile import gettempdir
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import requests
from requests.utils import DEFAULT_CA_BUNDLE_PATH

from supervisely.io import env as sly_env
from supervisely.io import fs as sly_fs
from supervisely.sly_logger import logger

random.seed(time.time())


def rand_str(length):
    chars = string.ascii_letters + string.digits  # [A-z][0-9]
    return "".join((random.choice(chars)) for _ in range(length))


def generate_free_name(used_names, possible_name, with_ext=False, extend_used_names=False):
    res_name = possible_name
    new_suffix = 1
    while res_name in set(used_names):
        if with_ext is True:
            res_name = "{}_{:02d}{}".format(
                sly_fs.get_file_name(possible_name),
                new_suffix,
                sly_fs.get_file_ext(possible_name),
            )
        else:
            res_name = "{}_{:02d}".format(possible_name, new_suffix)
        new_suffix += 1
    if extend_used_names:
        used_names.add(res_name)
    return res_name


def generate_names(base_name, count):
    name = sly_fs.get_file_name(base_name)
    ext = sly_fs.get_file_ext(base_name)

    names = [base_name]
    for idx in range(1, count):
        names.append("{}_{:02d}{}".format(name, idx, ext))

    return names


def camel_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_human(snake_str: str) -> str:
    """Return a human-readable string from a snake_case string.
    E.g. 'hello_world' -> 'Hello World'

    :param snake_str: snake_case string
    :type snake_str: str
    :return: Human-readable string
    :rtype: str
    """
    components = snake_str.split("_")
    return " ".join(word.capitalize() for word in components)


def take_with_default(v, default):
    return v if v is not None else default


def find_value_by_keys(d: Dict, keys: List[str], default=object()):
    for key in keys:
        if key in d:
            return d[key]
    if default is object():
        raise KeyError(f"None of the keys {keys} are in the dictionary.")
    return default


def batched(seq, batch_size=50):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def batched_iter(iterable, batch_size=50):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def get_bytes_hash(bytes):
    return base64.b64encode(hashlib.sha256(bytes).digest()).decode("utf-8")


def get_string_hash(data):
    return base64.b64encode(hashlib.sha256(str.encode(data)).digest()).decode("utf-8")


def unwrap_if_numpy(x):
    return x.item() if isinstance(x, np.number) else x


def _dprint(json_data):
    print(json.dumps(json_data))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


COMMUNITY = "community"
ENTERPRISE = "enterprise"


def validate_percent(value):
    if 0 <= value <= 100:
        pass
    else:
        raise ValueError("Percent has to be in range [0; 100]")


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)


def _remove_sensitive_information(d: dict):
    new_dict = copy.deepcopy(d)
    fields = ["api_token", "API_TOKEN", "AGENT_TOKEN", "apiToken", "spawnApiToken"]
    for field in fields:
        if field in new_dict:
            new_dict[field] = "***"

    for parent_key in ["state", "context"]:
        if parent_key in new_dict and type(new_dict[parent_key]) is dict:
            for field in fields:
                if field in new_dict[parent_key]:
                    new_dict[parent_key][field] = "***"
    return new_dict


def validate_img_size(img_size):
    if not isinstance(img_size, (tuple, list)):
        raise TypeError(
            '{!r} has to be a tuple or a list. Given type "{}".'.format("img_size", type(img_size))
        )
    return tuple(img_size)


def is_development() -> bool:
    mode = os.environ.get("ENV", "development")
    if mode == "production":
        return False
    else:
        return True


def is_debug_with_sly_net() -> bool:
    mode = os.environ.get("DEBUG_WITH_SLY_NET")
    if mode is not None:
        return True
    else:
        return False


def is_docker():
    path = "/proc/self/cgroup"
    return (
        os.path.exists("/.dockerenv")
        or os.path.isfile(path)
        and any("docker" in line for line in open(path))
    )


def is_production() -> bool:
    return not is_development()


def is_community() -> bool:
    server_address = sly_env.server_address()

    if (
        server_address.rstrip("/") == "https://app.supervise.ly"
        or server_address.rstrip("/") == "https://app.supervisely.com"
    ):
        return True
    else:
        return False


def abs_url(relative_url: str) -> str:
    from supervisely.api.api import SERVER_ADDRESS

    server_address = os.environ.get(SERVER_ADDRESS, "")
    if server_address == "":
        logger.warn("SERVER_ADDRESS env variable is not defined")
    return urllib.parse.urljoin(server_address, relative_url)


def compress_image_url(
    url: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    quality: Optional[int] = 70,
) -> str:
    """NOTE: This function is deprecated. Use resize_image_url instead.
    Returns a URL to a compressed image with given parameters.

    :param url: Full Image storage URL, can be obtained from ImageInfo.
    :type url: str
    :param width: Width of the compressed image.
    :type width: int, optional
    :param height: Height of the compressed image.
    :type height: int, optional
    :param quality: Quality of the compressed image.
    :type quality: int, optional
    :return: Full URL to a compressed image.
    :rtype: str
    """
    if width is None:
        width = ""
    if height is None:
        height = ""
    return url.replace(
        "/image-converter",
        f"/previews/{width}x{height},jpeg,q{quality}/image-converter",
    )


def resize_image_url(
    full_storage_url: str,
    ext: Literal["jpeg", "png"] = "jpeg",
    method: Literal["fit", "fill", "fill-down", "force", "auto"] = "auto",
    width: int = 0,
    height: int = 0,
    quality: int = 70,
) -> str:
    """Returns a URL to a resized image with given parameters.
    Default sizes are 0, which means that the image will not be resized,
    just compressed if the extension is jpeg to the given quality.
    Learn more about resize parameters `here <https://docs.imgproxy.net/usage/processing#resize>`_.

    :param full_storage_url: Full Image storage URL, can be obtained from ImageInfo.
    :type full_storage_url: str
    :param ext: Image extension, jpeg or png.
    :type ext: Literal["jpeg", "png"], optional
    :param method: Resize type, fit, fill, fill-down, force, auto.
    :type method: Literal["fit", "fill", "fill-down", "force", "auto"], optional
    :param width: Width of the resized image.
    :type width: int, optional
    :param height: Height of the resized image.
    :type height: int, optional
    :param quality: Quality of the resized image.
    :type quality: int, optional
    :return: Full URL to a resized image.
    :rtype: str

    :Usage example:

    .. code-block:: python

        import supervisely as sly
        from supervisely_utils import resize_image_url

        api = sly.Api(server_address, token)

        image_id = 376729
        img_info = api.image.get_info_by_id(image_id)

        img_resized_url = resize_image_url(
            img_info.full_storage_url, ext="jpeg", method="fill", width=512, height=256)
        print(img_resized_url)
        # Output: https://app.supervisely.com/previews/q/ext:jpeg/resize:fill:512:256:0/q:70/plain/h5un6l2bnaz1vj8a9qgms4-public/images/original/2/X/Re/<image_name>.jpg
    """
    # original url example: https://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/2/X/Re/<image_name>.jpg
    # resized url example:  https://app.supervisely.com/previews/q/ext:jpeg/resize:fill:300:0:0/q:70/plain/h5un6l2bnaz1vj8a9qgms4-public/images/original/2/X/Re/<image_name>.jpg
    # to add: previews/q/ext:jpeg/resize:fill:300:0:0/q:70/plain/
    try:
        parsed_url = urllib.parse.urlparse(full_storage_url)
        server_address = f"{parsed_url.scheme}://{parsed_url.netloc}"

        resize_string = f"previews/q/ext:{ext}/resize:{method}:{width}:{height}:0/q:{quality}/plain"
        url = full_storage_url.replace(server_address, f"{server_address}/{resize_string}")
        return url
    except Exception as e:
        logger.debug(f"Failed to resize image with url: {full_storage_url}: {repr(e)}")
        return full_storage_url


def get_storage_url(
    entity_type: Literal["dataset-entities", "dataset", "project", "file-storage"],
    entity_id: int,
    source_type: Literal["original", "preview"],
) -> str:
    """
    Generate URL for storage resources endpoints.

    :param entity_type: Type of entity ("dataset-entities", "dataset", "project", "file-storage")
    :type entity_type: str
    :param entity_id: ID of the entity
    :type entity_id: int
    :param source_type: Type of source ("original" or "preview")
    :type source_type: Literal["original", "preview"]
    :return: Storage URL
    :rtype: str
    """
    relative_url = f"/storage-resources/{entity_type}/{source_type}/{entity_id}"
    if is_development():
        return abs_url(relative_url)
    return relative_url


def get_image_storage_url(image_id: int, source_type: Literal["original", "preview"]) -> str:
    """
    Generate URL for image storage resources.

    :param image_id: ID of the image
    :type image_id: int
    :param source_type: Type of source ("original" or "preview")
    :type source_type: Literal["original", "preview"]
    :return: Storage URL for image
    :rtype: str
    """
    return get_storage_url("dataset-entities", image_id, source_type)


def get_dataset_storage_url(
    dataset_id: int, source_type: Literal["original", "preview", "raw"]
) -> str:
    """
    Generate URL for dataset storage resources.

    :param dataset_id: ID of the dataset
    :type dataset_id: int
    :param source_type: Type of source ("original", "preview", or "raw")
    :type source_type: Literal["original", "preview", "raw"]
    :return: Storage URL for dataset
    :rtype: str
    """
    return get_storage_url("dataset", dataset_id, source_type)


def get_project_storage_url(
    project_id: int, source_type: Literal["original", "preview", "raw"]
) -> str:
    """
    Generate URL for project storage resources.

    :param project_id: ID of the project
    :type project_id: int
    :param source_type: Type of source ("original", "preview", or "raw")
    :type source_type: Literal["original", "preview", "raw"]
    :return: Storage URL for project
    :rtype: str
    """
    return get_storage_url("project", project_id, source_type)


def get_file_storage_url(file_id: int) -> str:
    """
    Generate URL for file storage resources (raw files).

    :param file_id: ID of the file
    :type file_id: int
    :return: Storage URL for file
    :rtype: str
    """
    return get_storage_url("file-storage", file_id, "raw")


def get_preview_link(title="preview"):
    return (
        f'<a href="javascript:;">{title}<i class="zmdi zmdi-cast" style="margin-left: 5px"></i></a>'
    )


def get_datetime(value: str) -> datetime:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")


def get_readable_datetime(value: str) -> str:
    dt = get_datetime(value)
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_unix_timestamp() -> int:
    """Return the current Unix timestamp.

    :return: Current Unix timestamp.
    :rtype: int
    """
    return int(time.time())


def get_certificates_list(path: str = DEFAULT_CA_BUNDLE_PATH) -> List[str]:
    with open(path, "r", encoding="ascii") as f:
        content = f.read().strip()
        certs = []

        begin_cert = "-----BEGIN CERTIFICATE-----"
        end_cert = "-----END CERTIFICATE-----"

        while begin_cert in content:
            start_index = content.index(begin_cert)
            end_index = content.index(end_cert, start_index) + len(end_cert)
            cert = content[start_index:end_index]
            certs.append(cert)
            content = content[end_index:]
        return certs


def setup_certificates():
    """
    This function is used to add extra certificates to the default CA bundle on Supervisely import.
    """
    path_to_certificate: str = os.environ.get("SLY_EXTRA_CA_CERTS", "").strip()
    if path_to_certificate == "":
        return

    if os.path.exists(path_to_certificate):
        if os.path.isfile(path_to_certificate):
            with open(path_to_certificate, "r", encoding="ascii") as f:
                extra_ca_contents = f.read().strip()
                if extra_ca_contents == "":
                    raise RuntimeError(f"File with certificates is empty: {path_to_certificate}")

            certificates = get_certificates_list(DEFAULT_CA_BUNDLE_PATH)
            requests_ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE", "").strip()
            if requests_ca_bundle != "" and os.path.exists(requests_ca_bundle):
                if os.path.isfile(requests_ca_bundle):
                    certificates = get_certificates_list(requests_ca_bundle)
                else:
                    raise RuntimeError(f"Path to bundle is not a file: {requests_ca_bundle}")

            certificates.insert(0, extra_ca_contents)
            new_bundle_path = os.path.join(gettempdir(), "sly_extra_ca_certs.crt")
            with open(new_bundle_path, "w", encoding="ascii") as f:
                f.write("\n".join(certificates))

            old_request_ca_bundle_path = requests_ca_bundle
            os.environ["REQUESTS_CA_BUNDLE"] = new_bundle_path
            if (
                os.environ.get("SSL_CERT_FILE", "").strip() == ""
                or os.environ.get("SSL_CERT_FILE", "").strip() == old_request_ca_bundle_path
            ):
                os.environ["SSL_CERT_FILE"] = new_bundle_path
            logger.info(f"Certificates were added to the bundle: {path_to_certificate}")
        else:
            raise RuntimeError(f"Path to certificate is not a file: {path_to_certificate}")
    else:
        raise RuntimeError(f"Path to certificate does not exist: {path_to_certificate}")


def add_callback(func, callback):
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        callback()
        return res

    return wrapper


def compare_dicts(
    template: Dict[Any, Any], data: Dict[Any, Any], strict: bool = True
) -> Tuple[List[str], List[str]]:
    """Compare two dictionaries recursively (by keys only) and return lists of missing and extra fields.
    If strict is True, the keys of the template and data dictionaries must match exactly.
    Otherwise, the data dictionary may contain additional keys that are not in the template dictionary.

    :param template: The template dictionary.
    :type template: Dict[Any, Any]
    :param data: The data dictionary.
    :type data: Dict[Any, Any]
    :param strict: If True, the keys of the template and data dictionaries must match exactly.
    :type strict: bool, optional
    :return: A tuple containing a list of missing fields and a list of extra fields.
    :rtype: Tuple[List[str], List[str]]
    """
    missing_fields = []
    extra_fields = []

    if not isinstance(template, dict) or not isinstance(data, dict):
        return missing_fields, extra_fields

    if strict:
        template_keys = set(template.keys())
        data_keys = set(data.keys())

        missing_fields = list(template_keys - data_keys)
        extra_fields = list(data_keys - template_keys)

        for key in template_keys & data_keys:
            sub_missing, sub_extra = compare_dicts(template[key], data[key], strict)
            missing_fields.extend([f"{key}.{m}" for m in sub_missing])
            extra_fields.extend([f"{key}.{e}" for e in sub_extra])
    else:
        for key in template:
            if key not in data:
                missing_fields.append(key)
            else:
                sub_missing, sub_extra = compare_dicts(template[key], data[key], strict)
                missing_fields.extend([f"{key}.{m}" for m in sub_missing])
                extra_fields.extend([f"{key}.{e}" for e in sub_extra])

    return missing_fields, extra_fields


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get the current event loop or create a new one if it doesn't exist.
    Works for different Python versions and contexts.

    :return: Event loop
    :rtype: asyncio.AbstractEventLoop
    """
    try:
        # Preferred method for asynchronous context (Python 3.7+)
        return asyncio.get_running_loop()
    except RuntimeError:
        # If the loop is not running, get the current one or create a new one (Python 3.8 and 3.9)
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            # For Python 3.10+ or if the call occurs outside of an active loop context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


def run_coroutine(coroutine):
    """
    Runs an asynchronous coroutine in a synchronous context and waits for its result.

    This function checks if an event loop is already running:
    - If a loop is running, it schedules the coroutine using `asyncio.run_coroutine_threadsafe()`
      and waits for the result.
    - If no loop is running, it creates one and executes the coroutine with `run_until_complete()`.

    This ensures compatibility with both synchronous and asynchronous environments
    without creating unnecessary event loops.

    ⚠️ Note: This method is preferable when working with `asyncio` objects like `Semaphore`,
    since it avoids issues with mismatched event loops.

    :param coro: Asynchronous function.
    :type coro: Coroutine
    :return: Result of the asynchronous function.
    :rtype: Any

    :Usage example:

    .. code-block:: python

            from supervisely._utils import run_coroutine

            async def async_function():
                await asyncio.sleep(1)
                return "Hello, World!"

            coroutine = async_function()
            result = run_coroutine(coroutine)
            print(result)
            # Output: Hello, World!
    """

    loop = get_or_create_event_loop()

    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coroutine, loop=loop)
        return future.result()
    else:
        return loop.run_until_complete(coroutine)


def get_filename_from_headers(url):
    try:
        response = requests.head(url, allow_redirects=True)
        if response.status_code >= 400 or "Content-Disposition" not in response.headers:
            response = requests.get(url, stream=True)
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition:
            filename = re.findall('filename="?([^"]+)"?', content_disposition)
            if filename:
                return filename[0]
        filename = url.split("/")[-1] or "downloaded_file"
        return filename
    except Exception as e:
        print(f"Error retrieving file name from headers: {e}")
        return None


def get_valid_kwargs(kwargs, func, exclude=None):
    signature = inspect.signature(func)
    valid_kwargs = {}
    for key, value in kwargs.items():
        if exclude is not None and key in exclude:
            continue
        if key in signature.parameters:
            valid_kwargs[key] = value
    return valid_kwargs


def removesuffix(string, suffix):
    """
    Returns the string without the specified suffix if the string ends with that suffix.
    Otherwise returns the original string.
    Uses for Python versions < 3.9.

    :param string: The original string.
    :type string: str
    :param suffix: The suffix to remove.
    :type suffix: str
    :return: The string without the suffix or the original string.
    :rtype: str

    :Usage example:
    .. code-block:: python

        from supervisely._utils import removesuffix

        original_string = "example.txt"
        suffix_to_remove = ".txt"

        result = removesuffix(original_string, suffix_to_remove)
        print(result)

        # Output: example

    """
    if string.endswith(suffix):
        return string[: -len(suffix)]
    return string


def remove_non_printable(text: str) -> str:
    """Remove non-printable characters from a string.

    :param text: Input string
    :type text: str
    :return: String with non-printable characters removed
    :rtype: str
    """
    return "".join(char for char in text if char.isprintable()).strip()


def get_latest_instance_version_from_json() -> Optional[str]:
    """
    Get the latest (last) instance version from versions.json file.

    The versions.json file should contain a mapping of SDK versions to instance versions.
    This function returns the instance version from the last entry in the file.

    :return: Latest instance version or None if not found
    :rtype: Optional[str]
    """
    import json

    try:
        # Get the path to versions.json relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        versions_file = os.path.join(current_dir, "versions.json")

        if not os.path.exists(versions_file):
            logger.debug(f"versions.json file not found at {versions_file}")
            return None

        with open(versions_file, "r", encoding="utf-8") as f:
            versions_mapping = json.load(f)

        if not versions_mapping:
            return None

        # Get the last (latest) entry from the versions mapping
        # Since JSON preserves order in Python 3.7+, the last item is the latest
        latest_instance_version = list(versions_mapping.keys())[-1]
        logger.debug(f"Latest instance version found: {latest_instance_version}")
        return latest_instance_version

    except Exception:
        # Silently fail - don't break the import if versions.json is missing or malformed
        logger.debug("Failed to get latest instance version from versions.json")
        return None
