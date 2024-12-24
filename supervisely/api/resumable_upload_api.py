from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Coroutine, List, Optional

from supervisely._utils import camel_to_snake, get_or_create_event_loop
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
        response = await self._api.post_async(method, content=chunk, params=params)
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
