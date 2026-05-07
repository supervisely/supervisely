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
from tqdm import tqdm

from supervisely._utils import batched
from supervisely.api.module_api import ApiField, ModuleApi
from supervisely.imaging import image as sly_image
from supervisely.io.fs import ensure_base_path
from supervisely.sly_logger import logger


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

    async def async_stream_decoded(
        self,
        video_id: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """Stream decoded video frames as ``(frame_index, image)`` tuples via **PyAV**.

        Opens the remote video URL directly through PyAV (libav), decodes frames
        starting from frame 0, and yields only the requested range. Because each
        frame carries its real PTS (presentation timestamp), the method can detect
        and log gaps caused by corrupted or missing frames — the ``frame_index``
        emitted always reflects the *real* position of the frame in the video,
        not a sequential counter that would drift on missing frames.

        The method is correct for both **CFR** (constant frame rate) and **VFR**
        (variable frame rate) video, and works with any container format supported
        by libavformat (mp4, mkv, avi, webm, mov, …).

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param start: First frame index to stream (0-based, inclusive).
            Defaults to 0.
        :type start: int, optional
        :param end: Last frame index to stream (0-based, inclusive).
            Defaults to the last frame of the video.
        :type end: int, optional
        :returns: Async generator of ``(frame_index, np.ndarray)`` tuples where
            the image array has shape ``(H, W, 3)`` in **RGB** order.
        :rtype: AsyncGenerator[Tuple[int, np.ndarray], None]

        :raises RuntimeError: If ``fullStorageUrl`` is not available for the
            video, or if the container has no decodable video stream.

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
        # Resolve the full public URL of the video file.
        json_info = self._api.video.get_json_info_by_id(video_id, force_metadata_for_links=False)
        video_url = json_info.get("fullStorageUrl")
        if not video_url:
            raise RuntimeError(
                f"Cannot resolve fullStorageUrl for video {video_id}. "
                "Make sure the video is fully processed on the server."
            )

        # Replace the origin with the configured server address.
        parsed = urllib.parse.urlparse(video_url)
        public = urllib.parse.urlparse(self._api.server_address)
        video_url = urllib.parse.urlunparse(
            parsed._replace(scheme=public.scheme, netloc=public.netloc)
        )

        # Pass the API token as an HTTP request header to libavformat.
        av_options: Dict[str, str] = {}
        if self._api.token:
            av_options["headers"] = f"x-api-key: {self._api.token}\r\n"

        # Open the container in a thread-pool executor to avoid blocking the event loop.
        loop = asyncio.get_running_loop()
        container: av.container.InputContainer = await loop.run_in_executor(
            None,
            lambda: av.open(video_url, options=av_options),
        )

        try:
            video_streams = container.streams.video
            if not video_streams:
                raise RuntimeError(
                    f"PyAV found no video streams in video {video_id}."
                )
            v_stream = video_streams[0]

            time_base = v_stream.time_base  # e.g. Fraction(1, 90000)

            # Clamp the requested range.
            _start: int = max(0, start if start is not None else 0)
            _end: Optional[int] = max(_start, end) if end is not None else None

            decoded_idx: int = 0  # real frame position in the video
            prev_pts: Optional[int] = None
            prev_pts_gap: Optional[int] = None  # expected PTS step between consecutive frames

            for frame in container.decode(v_stream):
                current_pts: Optional[int] = frame.pts

                # Detect gaps: if the PTS jump is >1.5× the normal step, frames are missing.
                if current_pts is not None and prev_pts is not None:
                    gap = current_pts - prev_pts
                    if prev_pts_gap is not None and gap > prev_pts_gap * 1.5:
                        skipped = round(gap / prev_pts_gap) - 1
                        logger.warning(
                            "stream_frames: video_id=%d — detected gap of %d "
                            "missing frame(s) before decoded frame #%d "
                            "(PTS jumped from %d to %d, expected step ~%d).",
                            video_id,
                            skipped,
                            decoded_idx,
                            prev_pts,
                            current_pts,
                            prev_pts_gap,
                        )
                        # Advance the index to stay aligned with the real timeline.
                        decoded_idx += skipped
                    prev_pts_gap = gap
                prev_pts = current_pts

                if decoded_idx < _start:
                    decoded_idx += 1
                    continue

                if _end is not None and decoded_idx > _end:
                    break

                img: np.ndarray = frame.to_ndarray(format="rgb24")
                yield decoded_idx, img

                decoded_idx += 1

                # Yield control to the event loop on each frame.
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
        """Stream and save video frames to a local directory via **PyAV**.

        Wraps :meth:`async_stream_decoded` and writes each decoded frame to
        ``output_dir`` as ``frame_XXXXXX.<ext>``. The directory is created
        automatically if it does not exist. Images are saved in **RGB** order.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param output_dir: Directory where frames will be saved.
        :type output_dir: str
        :param start: First frame index to save (0-based, inclusive). Defaults to 0.
        :type start: int, optional
        :param end: Last frame index to save (0-based, inclusive). Defaults to last frame.
        :type end: int, optional
        :param ext: File extension without leading dot (e.g. ``"png"``, ``"jpg"``).
            Defaults to ``"png"``.
        :type ext: str, optional
        :param progress_cb: Optional callback called with ``1`` after each saved frame.
        :type progress_cb: tqdm or callable, optional
        :returns: List of absolute paths to saved frame files, in frame order.
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
        """Synchronous wrapper for :meth:`async_stream_to_dir`.
        Writes each decoded frame to
        ``output_dir`` as ``frame_XXXXXX.<ext>``. The directory is created
        automatically if it does not exist. Images are saved in **RGB** order.
        
        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param output_dir: Directory where frames will be saved.
        :type output_dir: str
        :param start: First frame index to save (0-based, inclusive). Defaults to 0.
        :type start: int, optional
        :param end: Last frame index to save (0-based, inclusive). Defaults to last frame.
        :type end: int, optional
        :param ext: File extension without leading dot (e.g. ``"png"``, ``"jpg"``).
            Defaults to ``"png"``.
        :type ext: str, optional
        :param progress_cb: Optional callback called with ``1`` after each saved frame.
        :type progress_cb: tqdm or callable, optional
        :returns: List of absolute paths to saved frame files, in frame order.
        :rtype: List[str]
        """
        return asyncio.run(
            self.async_stream_to_dir(video_id, output_dir, start, end, ext, progress_cb)
        )