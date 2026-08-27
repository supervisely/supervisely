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
import threading
import time
import urllib
import weakref
from collections import deque
from datetime import datetime
from functools import wraps
from tempfile import gettempdir
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple

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
    """
    Return a human-readable string from a snake_case string.
    E.g. 'hello_world' -> 'Hello World'

    :param snake_str: snake_case string
    :type snake_str: str
    :returns: Human-readable string
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
    """JSON encoder that converts NumPy scalars/arrays to built-in Python types for serialization."""

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
        logger.warning("SERVER_ADDRESS env variable is not defined")
    return urllib.parse.urljoin(server_address, relative_url)


def compress_image_url(
    url: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    quality: Optional[int] = 70,
) -> str:
    """NOTE: This function is deprecated. Use resize_image_url instead.
    Returns a URL to a compressed image with given parameters.

    :param url: Full Image storage URL, can be obtained from :class:`~supervisely.api.image_api.ImageInfo`.
    :type url: str
    :param width: Width of the compressed image.
    :type width: int, optional
    :param height: Height of the compressed image.
    :type height: int, optional
    :param quality: Quality of the compressed image.
    :type quality: int, optional
    :returns: Full URL to a compressed image.
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

    :param full_storage_url: Full Image storage URL, can be obtained from :class:`~supervisely.api.image_api.ImageInfo`.
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
    :returns: Full URL to a resized image.
    :rtype: str

    :Usage Example:

        .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly
            from supervisely_utils import resize_image_url

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            if sly.is_development():
                load_dotenv(os.path.expanduser("~/supervisely.env"))

            api = sly.Api.from_env()

            image_id = 376729
            img_info = api.image.get_info_by_id(image_id)

            img_resized_url = resize_image_url(
                img_info.full_storage_url,
                ext="jpeg",
                method="fill",
                width=512,
                height=256,
            )
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
    :returns: Storage URL
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
    :returns: Storage URL for image
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
    :returns: Storage URL for dataset
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
    :returns: Storage URL for project
    :rtype: str
    """
    return get_storage_url("project", project_id, source_type)


def get_file_storage_url(file_id: int) -> str:
    """
    Generate URL for file storage resources (raw files).

    :param file_id: ID of the file
    :type file_id: int
    :returns: Storage URL for file
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

    :returns: Current Unix timestamp.
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
    :returns: A tuple containing a list of missing fields and a list of extra fields.
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

    :returns: Event loop
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


def is_event_loop_running() -> bool:
    """
    Check whether an asyncio event loop is currently running in the calling thread.

    Unlike :func:`get_or_create_event_loop`, this function has no side effects: it never
    creates a loop. It is useful to decide whether it is safe to block on a coroutine via
    :func:`run_coroutine` (blocking on a loop that runs in the current thread would deadlock).

    :returns: True if an event loop is running in the current thread, False otherwise.
    :rtype: bool
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class _SemaphorePermitHolder:
    """
    Bookkeeping for permits of :class:`CrossLoopSemaphore` held by tasks of a single event loop.
    """

    __slots__ = ("_loop_ref", "count")

    def __init__(self, loop: asyncio.AbstractEventLoop):
        try:
            self._loop_ref = weakref.ref(loop)
        except TypeError:  # pragma: no cover - exotic loop implementations
            self._loop_ref = lambda bound=loop: bound
        self.count = 0

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop_ref()

    def is_dead(self) -> bool:
        """
        Whether the permits of this holder can never be released by their owner.

        A permit is released from a coroutine, so it requires a running event loop.
        If the loop is gone, closed or stopped, its tasks are frozen forever and
        the permits they hold are lost unless reclaimed.

        :returns: True if the permits are unreachable, False otherwise.
        :rtype: bool
        """
        loop = self._loop_ref()
        if loop is None or loop.is_closed():
            return True
        return not loop.is_running()


class CrossLoopSemaphore:
    """
    Counting semaphore that is not bound to a single asyncio event loop.

    ``asyncio.Semaphore`` binds itself to the event loop that first suspends on it and
    raises ``RuntimeError: ... is bound to a different event loop`` when awaited from
    another one. This makes it unusable as a process-wide request throttle: apps keep one
    long-lived :class:`~supervisely.api.api.Api` object but run work in freshly spawned
    threads (each thread gets its own event loop), so every run after the first fails.

    This implementation keeps the counter in plain Python guarded by a ``threading.Lock``
    and wakes waiters through ``loop.call_soon_threadsafe()`` on their own loop, so a
    single instance can be shared by any number of event loops and threads while still
    enforcing one global limit. The public interface matches ``asyncio.Semaphore``
    (``acquire()``, ``release()``, ``locked()``, ``async with``), so it is a drop-in
    replacement.

    Permits held by tasks of a loop that is closed or no longer running can never be
    released (the tasks are frozen), so they are reclaimed automatically when another
    acquirer would otherwise block forever. If such a task is resumed later and releases
    its permit, the release is dropped to keep the limit intact.

    :param value: Number of permits, i.e. the maximum number of concurrent holders.
    :type value: int

    :Usage Example:

     .. code-block:: python

        from supervisely._utils import CrossLoopSemaphore

        semaphore = CrossLoopSemaphore(5)

        async def download(url):
            async with semaphore:
                ...
    """

    def __init__(self, value: int = 1):
        if value < 0:
            raise ValueError("CrossLoopSemaphore initial value must be >= 0")
        self._limit = value
        self._value = value
        # permits that a shrinking resize() must swallow when they are returned
        self._debt = 0
        self._lock = threading.Lock()
        self._waiters: Deque[Tuple[asyncio.AbstractEventLoop, asyncio.Future]] = deque()
        self._holders: Dict[int, _SemaphorePermitHolder] = {}
        self.reclaimed_total = 0

    # ------------------------------------------------------------------ internals
    # every _*_locked() helper must be called with self._lock held

    def _holder_locked(self, loop: asyncio.AbstractEventLoop) -> _SemaphorePermitHolder:
        holder = self._holders.get(id(loop))
        if holder is None:
            holder = _SemaphorePermitHolder(loop)
            self._holders[id(loop)] = holder
        return holder

    def _held_locked(self) -> int:
        return sum(holder.count for holder in self._holders.values())

    def _return_permits_locked(self, count: int) -> None:
        paid = min(self._debt, count)
        self._debt -= paid
        self._value += count - paid

    def _reap_locked(self) -> int:
        """
        Reclaim permits held by frozen tasks. Returns the number of permits reclaimed.
        """
        reclaimed = 0
        for key, holder in list(self._holders.items()):
            if holder.is_dead():
                if holder.count > 0:
                    reclaimed += holder.count
                    self._return_permits_locked(holder.count)
                del self._holders[key]
        if reclaimed > 0:
            self.reclaimed_total += reclaimed
            logger.warning(
                f"Reclaimed {reclaimed} API semaphore permit(s) from tasks of an event loop "
                "that is no longer running. This means an async batch was abandoned "
                "(usually an exception in a gather) and its requests were never finished.",
                extra={"reclaimed_total": self.reclaimed_total, "semaphore_size": self._limit},
            )
            self._dispatch_locked()
        return reclaimed

    def _dispatch_locked(self) -> None:
        """
        Hand free permits to queued waiters: nothing else wakes them up.
        """
        while self._value > 0 and self._wake_one_locked():
            self._value -= 1

    def _drop_dead_waiters_locked(self) -> None:
        alive = deque()
        for loop, future in self._waiters:
            if loop.is_closed() or future.cancelled():
                continue
            alive.append((loop, future))
        self._waiters = alive

    @staticmethod
    def _grant(future: asyncio.Future) -> None:
        if not future.done():
            future.set_result(True)

    def _wake_one_locked(self) -> bool:
        """
        Hand one permit to the first live waiter. The permit stays accounted as held.
        """
        while self._waiters:
            loop, future = self._waiters.popleft()
            if future.cancelled() or loop.is_closed():
                continue
            holder = self._holder_locked(loop)
            holder.count += 1
            try:
                loop.call_soon_threadsafe(self._grant, future)
            except RuntimeError:
                # the loop was closed between the check above and this call
                holder.count -= 1
                if holder.count == 0:
                    self._holders.pop(id(loop), None)
                continue
            return True
        return False

    # -------------------------------------------------------------------- public
    async def acquire(self) -> bool:
        """
        Acquire a permit, waiting until one is available. Can be awaited from any loop.

        :returns: True when the permit is acquired.
        :rtype: bool
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._value <= 0:
                self._drop_dead_waiters_locked()
                self._reap_locked()
            # do not overtake waiters that are already queued
            if self._value > 0 and not self._waiters:
                self._value -= 1
                self._holder_locked(loop).count += 1
                return True
            future = loop.create_future()
            self._waiters.append((loop, future))
        try:
            await future
        except asyncio.CancelledError:
            with self._lock:
                for idx, (_, queued) in enumerate(self._waiters):
                    if queued is future:
                        del self._waiters[idx]
                        raise  # the permit was never granted, nothing to give back
            if future.done() and not future.cancelled():
                # cancelled after the permit was granted: hand it to the next waiter
                self.release()
            raise
        return True

    def release(self) -> None:
        """
        Release a permit, waking up the next waiter if there is one.

        Should be called from the same event loop that acquired the permit, which
        ``async with`` guarantees.
        """
        with self._lock:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            key = id(loop) if loop is not None else None
            holder = self._holders.get(key) if key is not None else None
            if holder is None:
                # release() from a sync context: attribute it only when unambiguous
                busy = [item for item in self._holders.items() if item[1].count > 0]
                if len(busy) == 1:
                    key, holder = busy[0]
            if holder is not None and holder.count > 0:
                holder.count -= 1
                if holder.count == 0:
                    del self._holders[key]
            if self._debt > 0:
                self._debt -= 1
                return
            if self._value + self._held_locked() >= self._limit + self._debt:
                # this permit was already reclaimed by _reap_locked(), or this is an
                # over-release: dropping it keeps the limit intact
                return
            if not self._wake_one_locked():
                self._value += 1

    def locked(self) -> bool:
        """
        Whether a permit cannot be acquired immediately.

        :returns: True if the semaphore cannot be acquired without waiting.
        :rtype: bool
        """
        with self._lock:
            return self._value <= 0 or bool(self._waiters)

    def resize(self, value: int) -> None:
        """
        Change the number of permits in place, keeping current holders and waiters.

        Growing releases queued waiters immediately. Shrinking takes effect as permits
        that are currently held are returned.

        :param value: New number of permits.
        :type value: int
        """
        if value < 0:
            raise ValueError("CrossLoopSemaphore size must be >= 0")
        with self._lock:
            diff = value - self._limit
            self._limit = value
            if diff > 0:
                paid = min(self._debt, diff)
                self._debt -= paid
                self._value += diff - paid
                self._dispatch_locked()
            elif diff < 0:
                taken = min(self._value, -diff)
                self._value -= taken
                self._debt += -diff - taken

    @property
    def limit(self) -> int:
        """
        Configured number of permits.

        :returns: Number of permits.
        :rtype: int
        """
        return self._limit

    def _state(self) -> Dict[str, int]:
        """
        Snapshot of the internal counters. For tests and debugging only.
        """
        with self._lock:
            held = self._held_locked()
            return {
                "limit": self._limit,
                "value": self._value,
                "held": held,
                "debt": self._debt,
                "waiters": len(self._waiters),
                "reclaimed": self.reclaimed_total,
                "consistent": self._value + held == self._limit + self._debt,
            }

    async def __aenter__(self) -> "CrossLoopSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __repr__(self) -> str:
        state = self._state()
        return (
            f"<CrossLoopSemaphore limit={state['limit']} free={state['value']} "
            f"held={state['held']} waiters={state['waiters']}>"
        )


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
    :returns: Result of the asynchronous function.
    :rtype: Any

    :Usage Example:

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
    :returns: The string without the suffix or the original string.
    :rtype: str

    :Usage Example:

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
    :returns: String with non-printable characters removed
    :rtype: str
    """
    return "".join(char for char in text if char.isprintable()).strip()


def get_latest_instance_version_from_json() -> Optional[str]:
    """
    Get the latest (last) instance version from versions.json file.

    The versions.json file should contain a mapping of SDK versions to instance versions.
    This function returns the instance version from the last entry in the file.

    :returns: Latest instance version or None if not found
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


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries. The override dictionary takes precedence over the base dictionary.
    - If a key exists in both dictionaries and both values are dicts, they are merged recursively.
    - In all other cases (including lists), the value from the override dictionary replaces the base value entirely.

    :param base: The base dictionary.
    :type base: dict
    :param override: The override dictionary.
    :type override: dict
    :returns: The merged dictionary.
    :rtype: dict
    """

    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
