# coding: utf-8
"""Work with video frames via the Supervisely API."""

# docs
from __future__ import annotations

import asyncio
import os
import re
import urllib.parse
from typing import AsyncGenerator, Callable, Dict, Generator, List, Optional, Tuple, Union

import av
import numpy as np
from requests_toolbelt import MultipartDecoder
from supervisely import logger
from tqdm import tqdm

from supervisely._utils import batched
from supervisely.api.module_api import ApiField, ModuleApi
from supervisely.imaging import image as sly_image
from supervisely.io.fs import ensure_base_path


class VideoFrameAPI(ModuleApi):
    """
    API for working with :class:`~supervisely.video_annotation.frame.Frame`.
    :class:`~supervisely.api.video.video_frame_api.VideoFrameAPI` object is immutable.
    """

    def _download(self, video_id: int, frame_index: int):
        """
        Private method. Download frame with given video ID and frame index.

        :param video_id: int
        :param frame_index: int
        :returns: Response class object containing frame data with given index from given video id
        """

        response = self._api.post(
            "videos.download-frame", {ApiField.VIDEO_ID: video_id, ApiField.FRAME: frame_index}
        )
        return response

    def _download_batch(
        self,
        video_id: int,
        frame_indexes: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ):
        """
        Private method. Batch download frames with given video ID and frame indexes.

        :param video_id: int
        :param frame_indexes: List[int]
        :returns: Response class object containing frame data with given index from given video id
        """

        for batch_ids in batched(frame_indexes):
            response = self._api.post(
                "videos.bulk.download-frame",
                {ApiField.VIDEO_ID: video_id, ApiField.FRAMES: batch_ids},
            )
            decoder = MultipartDecoder.from_response(response)
            for part in decoder.parts:
                content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                # The regex has 2 capture group: one for the prefix and one for the actual name value.
                frame_id = int(re.findall(r'(^|[\s;])name="(\d*)"', content_utf8)[0][1])

                if progress_cb is not None:
                    progress_cb(1)
                yield frame_id, part

    def download_np(self, video_id: int, frame_index: int) -> np.ndarray:
        """
        Download Image for frame with given index from given video ID in numpy format (RGB).

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_index: Index of frame to download.
        :type frame_index: int
        :returns: Image in RGB numpy matrix format
        :rtype: :class:`np.ndarray`

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                video_id = 198703211
                frame_idx = 5
                image_np = api.video.frame.download_np(video_id, frame_idx)
        """

        response = self._download(video_id, frame_index)
        frame = sly_image.read_bytes(response.content)
        return frame

    def download_nps(
        self,
        video_id: int,
        frame_indexes: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        keep_alpha: Optional[bool] = False,
    ) -> List[np.ndarray]:
        """
        Download frames with given indexes from given video ID in numpy format(RGB).

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_indexes: Indexes of frames to download.
        :type frame_indexes: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :returns: List of Images in RGB numpy matrix format
        :rtype: List[np.ndarray]

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                video_id = 198703211
                frame_indexes = [1,2,3,4,5,10,11,12,13,14,15]
                images_np = api.video.frame.download_nps(video_id, frame_indexes)
        """

        downloaded_frames = []
        for frame_bytes, frame_idx in zip(
            self.download_bytes(
                video_id=video_id, frame_indexes=frame_indexes, progress_cb=progress_cb
            ),
            frame_indexes,
        ):
            try:
                frame = sly_image.read_bytes(frame_bytes, keep_alpha)
                downloaded_frames.append(frame)
            except Exception as e:
                raise Exception(f"Couldn't read frame: {frame_idx}.") from e
        return downloaded_frames

    def download_nps_generator(
        self,
        video_id: int,
        frame_indexes: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        keep_alpha: Optional[bool] = False,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        for frame_idx, resp_part in self._download_batch(video_id, frame_indexes, progress_cb):
            frame_bytes = resp_part.content
            try:
                yield frame_idx, sly_image.read_bytes(frame_bytes, keep_alpha)
            except Exception as e:
                raise Exception(f"Couldn't read frame: {frame_idx}.") from e

    def download_path(self, video_id: int, frame_index: int, path: str) -> None:
        """
        Downloads frame on the given path for frame with given index from given Video ID.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_index: Index of frame to download.
        :type frame_index: int
        :param path: Local save path for image.
        :type path: str
        :returns: None
        :rtype: None

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                video_id = 198703211
                frame_idx = 5
                save_path = '/home/admin/Downloads/frames/result.png'
                api.video.frame.download_path(video_id, frame_idx, save_path)
        """

        response = self._download(video_id, frame_index)
        ensure_base_path(path)
        with open(path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

    def download_paths(
        self,
        video_id: int,
        frame_indexes: List[int],
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Downloads frames to given paths for frames with given indexes from given Video ID.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_indexes: Indexes of frames to download.
        :type frame_indexes: List[int]
        :param paths: Local save paths for frames.
        :type paths: List[str]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :returns: None
        :rtype: None

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                video_id = 198703211
                frame_indexes = [1,2,3,4,5,10,11,12,13,14,15]
                save_paths = [f"/home/admin/projects/video_project/frames/{idx}.png" for idx in frame_indexes]
                api.video.frame.download_paths(video_id, frame_indexes, save_paths)
        """

        if len(frame_indexes) == 0:
            return
        if len(frame_indexes) != len(paths):
            raise ValueError(
                'Can not match "indexes" and "paths" lists, len(frame_indexes) != len(paths)'
            )

        idx_to_path = {idx: path for idx, path in zip(frame_indexes, paths)}
        for frame_id, resp_part in self._download_batch(video_id, frame_indexes, progress_cb):
            with open(idx_to_path[frame_id], "wb") as w:
                w.write(resp_part.content)

    def download_bytes(
        self,
        video_id: int,
        frame_indexes: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[bytes]:
        """
        Download frames with given indexes from Dataset in Binary format.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_indexes: List of video frames indexes in Supervisely.
        :type frame_indexes: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :returns: List of Images in binary format
        :rtype: List[bytes]

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                video_id = 213542
                frame_indexes = [1,2,3,4,5,10,11,12,13,14,15]
                frames_bytes = api.video.frame.download_bytes(video_id=video_id, frame_indexes=frame_indexes)
                print(frames_bytes)
                # Output: [b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\\...']
        """

        if len(frame_indexes) == 0:
            return []

        idx_to_frame = {}
        for frame_idx, resp_part in self._download_batch(video_id, frame_indexes, progress_cb):
            idx_to_frame[frame_idx] = resp_part.content
        return [idx_to_frame[idx] for idx in frame_indexes]

    def _extract_video_url(self, json_info: dict, video_id: int) -> str:
        """Extract and rewrite the public video URL from already-fetched video JSON info."""
        video_url = json_info.get(ApiField.FULL_STORAGE_URL)
        if not video_url:
            raise RuntimeError(
                f"Cannot resolve fullStorageUrl for video {video_id}. "
                "Make sure the video is fully processed on the server."
            )
        parsed = urllib.parse.urlparse(video_url)
        public = urllib.parse.urlparse(self._api.server_address)
        return urllib.parse.urlunparse(
            parsed._replace(scheme=public.scheme, netloc=public.netloc)
        )

    def _resolve_video_url(self, video_id: int) -> str:
        """Resolve and rewrite the public video URL for the current server address."""
        json_info = self._api.video.get_json_info_by_id(video_id, force_metadata_for_links=False)
        return self._extract_video_url(json_info, video_id)

    def _av_open(self, video_url: str) -> "av.container.InputContainer":
        """Open an av container synchronously, injecting the API token header."""
        av_options: Dict[str, str] = {}
        if self._api.token:
            av_options["headers"] = f"x-api-key: {self._api.token}\r\n"
        return av.open(video_url, options=av_options)

    async def _async_build_pts_map(self, video_id: int) -> Tuple[List[int], bool]:
        """Build frame-index → PTS map and detect no-CTTS B-frame streams.

        Builds a map matching the web player's frame ordering by demuxing and
        filtering packets with negative PTS. Returns (pts_list, decode_from_start)
        where decode_from_start is True for B-frame videos without CTTS.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :returns: Tuple (pts_map, decode_from_start).
        :rtype: Tuple[List[int], bool]
        :raises RuntimeError: If the video URL cannot be resolved or the
            container contains no video stream.
        """
        video_url = self._resolve_video_url(video_id)
        loop = asyncio.get_running_loop()
        container: av.container.InputContainer = await loop.run_in_executor(
            None, lambda: self._av_open(video_url)
        )
        try:
            video_streams = container.streams.video
            if not video_streams:
                raise RuntimeError(f"PyAV found no video streams in video {video_id}.")
            v_stream = video_streams[0]

            pts_list: List[int] = []
            saw_pts_ne_dts = False
            for pkt in container.demux(v_stream):
                if pkt.pts is not None and pkt.dts is not None and pkt.pts != pkt.dts:
                    saw_pts_ne_dts = True
                if pkt.pts is not None and pkt.pts >= 0:  # matches mediabunny filter
                    pts_list.append(pkt.pts)

            pts_list.sort()  # matches mediabunny orderBy(samples, s => s.pts, 'asc')
            has_b_frames = getattr(v_stream.codec_context, "has_b_frames", 0) > 0
            decode_from_start = has_b_frames and not saw_pts_ne_dts
            if decode_from_start:
                logger.warning(
                    "B-frame stream without CTTS detected; "
                    "using full decode from start for stable frame extraction.",
                    extra={"video_id": video_id},
                )
            logger.debug(
                "build_pts_map: video_id=%d — %d pts values (demux, non-negative, sorted), decode_from_start=%s.",
                video_id,
                len(pts_list),
                decode_from_start,
            )
            return pts_list, decode_from_start
        finally:
            container.close()

    async def async_stream_decoded(
        self,
        video_id: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """Stream decoded video frames by frame index.

        Builds a PTS map by demuxing, then decodes the requested frame range.
        Automatically handles no-CTTS B-frame videos by decoding from stream start.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param start: First frame index (0-based, inclusive). Defaults to 0.
        :type start: int, optional
        :param end: Last frame index (0-based, inclusive). Defaults to last frame.
        :type end: int, optional
        :returns: Async generator of (frame_index, image_array) tuples.
                  Image array shape is (H, W, 3) in RGB order.
        :rtype: AsyncGenerator[Tuple[int, np.ndarray], None]

        :Usage Example:

            .. code-block:: python

                import asyncio
                import supervisely as sly

                api = sly.Api.from_env()

                async def main():
                    # Frames 150..380 (inclusive)
                    async for frame_idx, img in api.video.frame.stream_frames(
                        video_id=19371139, start=150, end=380
                    ):
                        print(frame_idx, img.shape)  # e.g. 150 (1080, 1920, 3)

                asyncio.run(main())
        """
        # --- Phase 1: build PTS map and detect fallback mode. -------------------
        pts_map, decode_from_start = await self._async_build_pts_map(video_id)

        if not pts_map:
            return

        _start: int = max(0, start if start is not None else 0)
        _end: int = min(max(_start, end), len(pts_map) - 1) if end is not None else len(pts_map) - 1

        # Reverse map: CTS value → frame index (in mediabunny order).
        pts_to_index: Dict[int, int] = {pts: idx for idx, pts in enumerate(pts_map)}

        # --- Phase 2: open a fresh connection. ----------------------------------
        video_url = self._resolve_video_url(video_id)
        loop = asyncio.get_running_loop()
        container: av.container.InputContainer = await loop.run_in_executor(
            None, lambda: self._av_open(video_url)
        )
        try:
            video_streams = container.streams.video
            if not video_streams:
                raise RuntimeError(f"PyAV found no video streams in video {video_id}.")
            v_stream = video_streams[0]

            if decode_from_start:
                # no-CTTS fallback: decode packets from the beginning in DTS order.
                # This avoids seek/reorder artifacts on rare B-frame streams.
                remaining: set = set(range(_start, _end + 1))
                for pkt in container.demux(v_stream):
                    pkt_idx: Optional[int] = (
                        pts_to_index.get(pkt.pts)
                        if pkt.pts is not None and pkt.pts >= 0
                        else None
                    )
                    is_target = pkt_idx is not None and _start <= pkt_idx <= _end

                    output_frame: Optional["av.VideoFrame"] = None
                    for frame in pkt.decode():
                        if frame.pts is not None and not frame.is_corrupt:
                            output_frame = frame
                            break

                    if is_target and output_frame is not None:
                        yield pkt_idx, output_frame.to_ndarray(format="rgb24")
                        remaining.discard(pkt_idx)
                        await asyncio.sleep(0)

                    if not remaining:
                        break
                return

            for frame_idx in range(_start, _end + 1):
                target_pts: int = pts_map[frame_idx]

                # Seek to the nearest keyframe at or before target.
                container.seek(target_pts, stream=v_stream, backward=True, any_frame=False)

                closest_frame: Optional["av.VideoFrame"] = None
                closest_dist: Optional[int] = None

                for frame in container.decode(v_stream):
                    if frame.pts is None or frame.is_corrupt:
                        continue

                    if frame.pts not in pts_to_index:
                        continue

                    dist = abs(frame.pts - target_pts)
                    if closest_dist is None or dist < closest_dist:
                        closest_frame = frame
                        closest_dist = dist
                    elif frame.pts > target_pts:
                        break

                if closest_frame is not None:
                    yield frame_idx, closest_frame.to_ndarray(format="rgb24")
                    await asyncio.sleep(0)
        finally:
            container.close()

    async def async_stream_to_dir(
        self,
        video_id: int,
        output_dir: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        ext: str = "png",
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[str]:
        """Stream and save video frames to disk.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param output_dir: Directory where frames will be saved.
        :type output_dir: str
        :param start: First frame index (0-based, inclusive).
        :type start: int, optional
        :param end: Last frame index (0-based, inclusive).
        :type end: int, optional
        :param ext: File extension without dot. Defaults to "png".
        :type ext: str, optional
        :param progress_cb: Optional callback(1) after each saved frame.
        :type progress_cb: tqdm or callable, optional
        :returns: List of saved file paths in frame order.
        :rtype: List[str]

        :Usage Example:

            .. code-block:: python

                import asyncio
                import supervisely as sly

                api = sly.Api.from_env()

                async def main():
                    paths = await api.video.frame.stream_frames_to_dir(
                        video_id=19371139,
                        output_dir="./frames",
                        start=0,
                        end=99,
                    )
                    print(paths[0])  # ./frames/frame_000000.png

                asyncio.run(main())
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths: List[str] = []
        async for frame_idx, img in self.async_stream_decoded(video_id, start=start, end=end):
            path = os.path.join(output_dir, f"frame_{frame_idx:06d}.{ext}")
            sly_image.write(path, img)
            saved_paths.append(path)
            if progress_cb is not None:
                progress_cb(1)
        return saved_paths

    def stream_to_dir(
        self,
        video_id: int,
        output_dir: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        ext: str = "png",
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[str]:
        """Stream and save video frames to disk. 
        Synchronous wrapper for `async_stream_to_dir`.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param output_dir: Directory where frames will be saved.
        :type output_dir: str
        :param start: First frame index (0-based, inclusive).
        :type start: int, optional
        :param end: Last frame index (0-based, inclusive).
        :type end: int, optional
        :param ext: File extension without dot. Defaults to "png".
        :type ext: str, optional
        :param progress_cb: Optional callback(1) after each saved frame.
        :type progress_cb: tqdm or callable, optional
        :returns: List of saved file paths in frame order.
        :rtype: List[str]
        """
        return asyncio.run(
            self.async_stream_to_dir(video_id, output_dir, start, end, ext, progress_cb)
        )