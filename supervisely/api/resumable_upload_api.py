from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncGenerator, Coroutine, List, Optional, Tuple
from uuid import uuid4

from aiofiles.threadpool.binary import AsyncBufferedReader

from supervisely._utils import camel_to_snake
from supervisely.api.module_api import ApiField

if TYPE_CHECKING:
    from supervisely.api.api import Api


@dataclass
class Limits:
    is_parallel_upload_supported: bool
    min_chunk_size: int
    max_chunk_size: int
    max_chunks: int
    chunk_size_multiple_of: int

    def calculate_optimal_chunk_size(self, file_size: int) -> Tuple[int, int]:
        """Calculate optimal chunk size based on limits.

        :param file_size: Size of the file in bytes.
        :type file_size: int
        :return: Optimal chunk size and number of chunks.
        :rtype: Tuple[int, int]
        """
        if self.min_chunk_size is None:
            self.min_chunk_size = 1024 * 1024  # 1 MB
        optimal_chunk_size = min(file_size // self.max_chunks, self.max_chunk_size)
        optimal_chunk_size = max(optimal_chunk_size, self.min_chunk_size)
        num_chunks = (file_size + optimal_chunk_size - 1) // optimal_chunk_size
        return optimal_chunk_size, num_chunks


@dataclass
class ResumableResponse:
    session_id: str
    limits: Limits


def transform_keys(obj):
    """Recursively transform keys of a dictionary from camelCase to snake_case."""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
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
        session_id = response_json[ApiField.SESSION_ID]
        limits = response_json[ApiField.LIMITS]
        limits = Limits(**transform_keys(limits))
        return ResumableResponse(session_id=session_id, limits=limits)

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
        return self.parse_resumable_response(response.json())

    async def upload_chunk(
        self,
        chunk: bytes,
        team_id: int,
        file_path: str,
        session_id: str,
        part_id: int,
        offset: float,
    ) -> Coroutine[ResumableResponse]:
        """Upload a chunk of the file to the storage.

        :param chunk: Chunk data as a byte string.
        :type chunk: bytes
        :param team_id: Team ID.
        :type team_id: int
        :param file_path: Path to the file in the storage.
        :type file_path: str
        :param session_id: Session ID.
        :type session_id: str
        :param part_id: Part ID.
        :type part_id: int
        :param offset: Offset in bytes to start upload from.
        :type offset: float
        :return: Resumable upload response.
        :rtype: ResumableResponse
        """
        method = "file-storage.resumable_upload.chunk"
        params = {
            ApiField.TEAM_ID: team_id,
            ApiField.PATH: file_path,
            ApiField.SESSION_ID: session_id,
            ApiField.PART_ID: part_id,
            ApiField.OFFSET: offset,
        }
        headers = {"Content-Type": "application/octet-stream"}
        response = await self._api.post_async(method, content=chunk, params=params, headers=headers)
        return self.parse_resumable_response(response.json())

    def complete_upload(
        self,
        team_id: int,
        file_path: str,
        session_id: str,
        parts: Optional[List[int]] = None,
    ) -> ResumableResponse:
        """Complete the upload process.

        :param team_id: Team ID.
        :type team_id: int
        :param file_path: Path to the file in the storage.
        :type file_path: str
        :param session_id: Session ID.
        :type session_id: str
        :param parts: List of part partIds in the order they should be concatenated. This information is required for parallel uploads.
        :type parts: List[int]
        :return: Resumable upload response.
        :rtype: ResumableResponse
        """
        method = "file-storage.resumable_upload.complete"
        data = {
            ApiField.TEAM_ID: team_id,
            ApiField.PATH: file_path,
            ApiField.SESSION_ID: session_id,
            ApiField.PARTS: parts or [],
        }
        response = self._api.post_httpx(method, json=data)
        return self.parse_resumable_response(response.json())

    def abort_upload(
        self,
        team_id: int,
        file_path: str,
        session_id: str,
    ) -> ResumableResponse:
        """Abort the upload process.

        :param team_id: Team ID.
        :type team_id: int
        :param file_path: Path to the file in the storage.
        :type file_path: str
        :param session_id: Session ID.
        :type session_id: str
        :return: Resumable upload response.
        :rtype: ResumableResponse
        """
        method = "file-storage.resumable_upload.abort"
        data = {
            ApiField.TEAM_ID: team_id,
            ApiField.PATH: file_path,
            ApiField.SESSION_ID: session_id,
        }
        response = self._api.post_httpx(method, json=data)
        return self.parse_resumable_response(response.json())

    def get_upload_status(
        self,
        team_id: int,
        file_path: str,
        session_id: str,
    ) -> ResumableResponse:
        """Retrieve the status of the upload process.

        :param team_id: Team ID.
        :type team_id: int
        :param file_path: Path to the file in the storage.
        :type file_path: str
        :param session_id: Session ID.
        :type session_id: str
        :return: Resumable upload response.
        :rtype: ResumableResponse
        """
        method = "file-storage.resumable_upload.status"
        data = {
            ApiField.TEAM_ID: team_id,
            ApiField.PATH: file_path,
            ApiField.SESSION_ID: session_id,
        }
        response = self._api.post_httpx(method, json=data)
        return self.parse_resumable_response(response.json())

    @staticmethod
    def generate_id() -> str:
        """Generate a unique identifier for the upload part.

        :return: Unique identifier as a hexadecimal string from UUID4.
        :rtype: str
        """
        return uuid4().hex

    @staticmethod
    async def generate_chunks(
        fd: AsyncBufferedReader,
        chunk_size: int,
        num_chunks: int,
    ) -> AsyncGenerator[bytes, None]:
        """Generate chunks of data from a file descriptor.

        :param fd: File descriptor.
        :type fd: AsyncBufferedReader
        :param chunk_size: Size of the chunk in bytes.
        :type chunk_size: int
        :param num_chunks: Number of chunks to generate.
        :type num_chunks: int
        :return: Asynchronous generator of chunks.
        :rtype: AsyncGenerator[bytes, None]
        """
        for _ in range(num_chunks):
            chunk = await fd.read(chunk_size)
            if not chunk:
                break
            yield chunk
