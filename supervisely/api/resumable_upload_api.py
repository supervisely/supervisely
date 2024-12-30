from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    Callable,
    Coroutine,
    List,
    Optional,
    Tuple,
)

from aiofiles.threadpool.binary import AsyncBufferedReader

from supervisely import logger
from supervisely._utils import camel_to_snake
from supervisely.api.module_api import ApiField

if TYPE_CHECKING:
    from supervisely.api.api import Api
    from supervisely.api.file_api import FileInfo


@dataclass
class Limits:
    is_parallel_upload_supported: bool
    min_chunk_size: int
    max_chunk_size: int
    max_chunks: int
    chunk_size_multiple_of: int
    optimal_chunk_size: Optional[int] = None
    num_chunks: Optional[int] = None

    def __post_init__(self):
        self.min_chunk_size = int(self.min_chunk_size) if self.min_chunk_size is not None else None
        self.max_chunk_size = int(self.max_chunk_size) if self.max_chunk_size is not None else None
        self.max_chunks = int(self.max_chunks) if self.max_chunks is not None else None

    def calculate_optimal_chunk_size(self, file_size: int) -> Limits:
        """Calculate optimal chunk size based on limits.

        :param file_size: Size of the file in bytes.
        :type file_size: int
        :return: Limits object with optimal_chunk_size and num_chunks set.
        :rtype: Limits
        """
        if self.min_chunk_size is None:
            self.min_chunk_size = 1024 * 1024  # 1 MB
        optimal_chunk_size = min(file_size // self.max_chunks, self.max_chunk_size)
        optimal_chunk_size = max(optimal_chunk_size, self.min_chunk_size)

        # for google cloud storage
        if self.chunk_size_multiple_of:
            optimal_chunk_size = (
                optimal_chunk_size // self.chunk_size_multiple_of
            ) * self.chunk_size_multiple_of

        num_chunks = (file_size + optimal_chunk_size - 1) // optimal_chunk_size
        if num_chunks == 1:
            optimal_chunk_size = file_size
        self.optimal_chunk_size = optimal_chunk_size
        self.num_chunks = num_chunks
        return self


@dataclass
class Range:
    start: int
    end: int

    def __post_init__(self):
        self.start = int(self.start) if self.start is not None else None
        self.end = int(self.end) if self.end is not None else None


@dataclass
class Part:
    part_id: int
    size: int
    range: Range

    def __post_init__(self):
        self.part_id = int(self.part_id)
        self.size = int(self.size)


@dataclass
class ResumableStatus:
    parts: Optional[List[Part]] = None

    def is_uploaded(self, limits: Optional[Limits] = None) -> bool:
        """Check if all parts have been uploaded. Based on the number of chunks in the file.

        :param limits: Limits object with num_chunks information.
        :type limits: Optional[Limits]
        :return: True if all parts have been uploaded, False otherwise.
        :rtype: bool
        """
        num_chunks = limits.num_chunks if limits is not None else None
        missing_ids = ResumableResponse(status=self).get_part_ids_to_reupload(num_chunks)
        if missing_ids:
            return False
        else:
            return True


@dataclass
class ResumableResponse:
    team_id: Optional[int] = None
    file_path: Optional[str] = None
    session_id: Optional[str] = None
    limits: Optional[Limits] = None
    hash: Optional[str] = None
    status: Optional[ResumableStatus] = None
    file_info: Optional[FileInfo] = None

    def get_part_ids_to_reupload(self, num_chunks: Optional[int] = None) -> List[int]:
        """Get a list of Parts that need to be re-uploaded.

        :param num_chunks: Number of chunks in the file.
        :type num_chunks: Optional[int]
        :return: List of Parts.
        :rtype: List[Part]
        :raises ValueError: If the number of chunks is not provided and cannot be determined.
        """
        if num_chunks is None:
            num_chunks = self.limits.num_chunks
            if num_chunks is None:
                raise ValueError("Cannot determine the number of chunks in the file.")
        if self.status is None or self.status.parts is None:
            return []
        else:
            part_ids = sorted(part.part_id for part in self.status.parts)
            missing_ids = [i for i in range(part_ids[0], num_chunks + 1) if i not in part_ids]
            return missing_ids


def transform_keys(obj):
    """Recursively transform keys of a dictionary from camelCase to snake_case."""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == ApiField.RANGE:
                new_obj[ApiField.RANGE] = Range(**transform_keys(v))
            else:
                new_obj[camel_to_snake(k)] = transform_keys(v)
        return new_obj
    elif isinstance(obj, list):
        return [transform_keys(i) for i in obj]
    else:
        return obj


class ResumableUploadApi:

    def __init__(self, api: "Api"):
        self._api = api

    @staticmethod
    def parse_resumable_response(response_json: dict) -> ResumableResponse:
        session_id = response_json.get(ApiField.SESSION_ID)
        limits = response_json.get(ApiField.LIMITS)
        hash = response_json.get(ApiField.HASH)
        if limits is not None:
            limits = Limits(**transform_keys(limits))
        return ResumableResponse(session_id=session_id, limits=limits, hash=hash)

    @staticmethod
    def parse_resumable_status(response_json: dict) -> ResumableStatus:
        parts = response_json.get(ApiField.PARTS)
        if parts is not None:
            parts = [Part(**transform_keys(part)) for part in parts]
        return ResumableStatus(parts=parts)

    def request_upload(
        self,
        team_id: int,
        file_path: str,
        size: Optional[int] = None,
        sha256: Optional[str] = None,
        crc32: Optional[str] = None,
        blake3: Optional[str] = None,
    ) -> ResumableResponse:
        """Initialize a resumable upload.
        Returns information about the upload process: a session ID and limits for the upload.

        :param team_id: Team ID.
        :type team_id: int
        :param file_path: Path to the file in the storage.
        :type file_path: str
        :param size: Size of the file in bytes.
        :type size: int
        :param sha256: SHA-256 hash of the file.
        :type sha256: str
        :param crc32: CRC32 hash of the file.
        :type crc32: str
        :param blake3: BLAKE3 hash of the file.
        :type blake3: str
        :return: Resumable upload response.
        :rtype: ResumableResponse
        """
        method = "file-storage.resumable_upload.request"
        data = {ApiField.TEAM_ID: team_id, ApiField.PATH: file_path, ApiField.META: {}}
        if size is not None:
            data[ApiField.META][ApiField.SIZE] = size
        if sha256 is not None:
            data[ApiField.META][ApiField.SHA256] = sha256
        if crc32 is not None:
            data[ApiField.META][ApiField.CRC32] = crc32
        if blake3 is not None:
            data[ApiField.META][ApiField.BLAKE3] = blake3
        response = self._api.post_httpx(method, json=data)
        resumable_response = self.parse_resumable_response(response.json())
        resumable_response.team_id = team_id
        resumable_response.file_path = file_path
        return resumable_response

    async def upload_chunk(
        self,
        chunk: bytes,
        session_id: str,
        part_id: int,
        offset: float,
        semaphore: Optional[asyncio.Semaphore] = None,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Coroutine[ResumableResponse]:
        """Upload a chunk of the file to the storage.
        In response server returns hash of the uploaded chunk.

        :param chunk: Chunk data as a byte string.
        :type chunk: bytes
        :param team_id: Team ID.
        :type team_id: int
        :param part_id: Part ID.
        :type part_id: int
        :param offset: Offset in bytes to start upload from.
        :type offset: float
        :param semaphore: Semaphore to limit the number of concurrent uploads.
        :type semaphore: Optional[asyncio.Semaphore]
        :param progress_cb: Callback function to report progress in bytes.
        :type progress_cb: Optional[Callable[[int], None]]
        :return: Resumable upload response.
        :rtype: ResumableResponse
        """
        method = "file-storage.resumable_upload.chunk"
        params = {
            ApiField.SESSION_ID: session_id,
            ApiField.PART_ID: part_id,
            ApiField.OFFSET: offset,
        }
        headers = {"Content-Type": "application/octet-stream"}

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        async with semaphore:
            response = await self._api.post_async(
                method, content=chunk, params=params, headers=headers
            )
        if progress_cb is not None:
            progress_cb(len(chunk))
        return self.parse_resumable_response(response.json())

    def complete_upload(
        self,
        session_id: str,
        parts: Optional[List[int]] = None,
    ) -> FileInfo:
        """Complete the upload process and create a file in the storage.
        Returns information about the uploaded file.

        :param session_id: Session ID.
        :type session_id: str
        :param parts: List of part partIds in the order they should be concatenated.
                    This information is required for parallel uploads.
        :type parts: List[int]
        :return: File information.
        :rtype: FileInfo
        """
        method = "file-storage.resumable_upload.complete"
        data = {
            ApiField.SESSION_ID: session_id,
            ApiField.PARTS: parts or [],
        }
        response = self._api.post_httpx(method, json=data)
        return self._api.file._convert_json_info(response.json())

    def abort_upload(
        self,
        session_id: str,
    ) -> bool:
        """Abort the upload process.

        :param session_id: Session ID.
        :type session_id: str
        :return: True if the upload was successfully aborted, False otherwise.
        """

        method = "file-storage.resumable_upload.abort"
        data = {ApiField.SESSION_ID: session_id}
        try:
            response = self._api.post_httpx(method, json=data)
        except Exception as e:
            logger.error(f"Failed to abort upload for session '{session_id}'", exc_info=e)
            result = False
        else:
            result = response.json().get("success", False)
        return result

    def get_upload_status(
        self,
        session_id: str,
    ) -> ResumableStatus:
        """Retrieve the status of the upload process.
        Status includes information about uploaded parts.

        :param session_id: Session ID.
        :type session_id: str
        :return: Resumable upload status response.
        :rtype: ResumableStatus
        """
        method = "file-storage.resumable_upload.status"
        data = {ApiField.SESSION_ID: session_id}
        response = self._api.post_httpx(method, json=data)
        return self.parse_resumable_status(response.json())

    @staticmethod
    async def generate_chunks(
        fd: AsyncBufferedReader,
        limits: Limits,
    ) -> AsyncGenerator[bytes, None]:
        """Generate chunks of data from a file descriptor.

        :param fd: File descriptor.
        :type fd: AsyncBufferedReader
        :return: Asynchronous generator of chunks.
        :rtype: AsyncGenerator[bytes, None]
        """
        for _ in range(limits.num_chunks):
            chunk = await fd.read(limits.optimal_chunk_size)
            if not chunk:
                break
            yield chunk
