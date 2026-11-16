"""Resuming a video project download into a folder that already holds some of its videos.

With ``resume_download=True`` the sync ``download_video_project`` added every video again and
raised ``Item '...' already exists in dataset`` on the first one already there. That is the
fallback ``download_to_cache`` takes when the async download fails partway, so every later run
failed the same way. A resumed download now keeps a video whose saved info and file size match
the server, and replaces one that does not.
"""

import asyncio
import os
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project import OpenMode
from supervisely.project.video_project import (
    VideoProject,
    download_video_project,
    download_video_project_async,
)
from supervisely.video_annotation.video_annotation import VideoAnnotation

PROJECT_ID = 1
VideoInfo = namedtuple("VideoInfo", ["id", "name", "updated_at", "file_meta", "custom_data"])


def _content(video_id):
    return bytes([video_id % 256]) * (1000 + video_id)


def _video(video_id, updated_at="2026-01-01T00:00:00.000Z", with_size=True):
    return VideoInfo(
        id=video_id,
        name=f"clip_{video_id}.mp4",
        updated_at=updated_at,
        file_meta={"size": str(len(_content(video_id)))} if with_size else {},
        custom_data={},
    )


def _ann_json(video):
    ann = VideoAnnotation((240, 320), 10).to_json()
    ann["videoId"] = video.id
    ann["videoName"] = video.name
    return ann


class _FakeApi:
    def __init__(self, videos):
        self.videos = videos
        self.downloaded = []
        dataset = SimpleNamespace(id=10, name="ds")
        self.project = MagicMock()
        self.project.get_meta.return_value = ProjectMeta().to_json()
        self.dataset = MagicMock()
        self.dataset.tree.side_effect = lambda project_id: iter([([], dataset)])
        self.video = MagicMock()
        self.video.get_list.side_effect = lambda dataset_id: list(self.videos)
        by_id = lambda: {v.id: v for v in self.videos}
        self.video.annotation.download_bulk.side_effect = lambda ds_id, ids: [
            _ann_json(by_id()[i]) for i in ids
        ]

        async def ann_async(video_id, video, semaphore=None):
            return [_ann_json(video)]

        self.video.annotation.download_async.side_effect = ann_async
        self.video.download_path.side_effect = self._download

        async def download_async(video_id, path, semaphore=None, **kwargs):
            self._download(video_id, path)

        self.video.download_path_async.side_effect = download_async

    def _download(self, video_id, path, *args, **kwargs):
        self.downloaded.append(video_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(_content(video_id))

    def get_default_semaphore(self):
        return asyncio.Semaphore(4)


def _download(api, dest_dir, use_async, **kwargs):
    kwargs.setdefault("save_video_info", True)
    kwargs.setdefault("log_progress", False)
    if use_async:
        asyncio.run(download_video_project_async(api, PROJECT_ID, dest_dir, **kwargs))
    else:
        download_video_project(api, PROJECT_ID, dest_dir, **kwargs)


def _assert_matches_server(dest_dir, videos):
    ds = VideoProject(dest_dir, OpenMode.READ).datasets.get("ds")
    assert sorted(ds.get_items_names()) == sorted(v.name for v in videos)
    for v in videos:
        with open(ds.get_item_path(v.name), "rb") as f:
            assert f.read() == _content(v.id)
        assert ds.get_item_info(v.name).updated_at == v.updated_at
        assert os.path.isfile(ds.get_ann_path(v.name))


@pytest.mark.parametrize("use_async", [False, True])
def test_resume_keeps_matching_videos_and_adds_new_ones(tmp_path, use_async):
    dest_dir = str(tmp_path / "project")
    api = _FakeApi([_video(1), _video(2)])
    _download(api, dest_dir, use_async)

    api.videos.append(_video(3))
    api.downloaded.clear()
    _download(api, dest_dir, use_async, resume_download=True)

    assert api.downloaded == [3]
    _assert_matches_server(dest_dir, api.videos)


@pytest.mark.parametrize("use_async", [False, True])
def test_resume_replaces_stale_and_truncated_videos(tmp_path, use_async):
    dest_dir = str(tmp_path / "project")
    api = _FakeApi([_video(1), _video(2), _video(3)])
    _download(api, dest_dir, use_async)

    ds = VideoProject(dest_dir, OpenMode.READ).datasets.get("ds")
    with open(ds.get_item_path("clip_2.mp4"), "r+b") as f:
        f.truncate(100)
    api.videos[0] = _video(1, updated_at="2026-02-01T00:00:00.000Z")
    api.downloaded.clear()
    _download(api, dest_dir, use_async, resume_download=True)

    assert sorted(api.downloaded) == [1, 2]
    _assert_matches_server(dest_dir, api.videos)


@pytest.mark.parametrize("use_async", [False, True])
def test_resume_without_saved_info_downloads_again(tmp_path, use_async):
    dest_dir = str(tmp_path / "project")
    api = _FakeApi([_video(1), _video(2)])
    _download(api, dest_dir, use_async, save_video_info=False)

    api.downloaded.clear()
    _download(api, dest_dir, use_async, save_video_info=False, resume_download=True)

    assert sorted(api.downloaded) == [1, 2]
    ds = VideoProject(dest_dir, OpenMode.READ).datasets.get("ds")
    assert sorted(ds.get_items_names()) == ["clip_1.mp4", "clip_2.mp4"]


@pytest.mark.parametrize("use_async", [False, True])
def test_resume_does_not_keep_placeholders_when_size_is_unknown(tmp_path, use_async):
    """Placeholders from a run without videos are not kept when the server reports no size."""
    dest_dir = str(tmp_path / "project")
    api = _FakeApi([_video(1, with_size=False), _video(2, with_size=False)])
    _download(api, dest_dir, use_async, download_videos=False)

    api.downloaded.clear()
    _download(api, dest_dir, use_async, resume_download=True)

    assert sorted(api.downloaded) == [1, 2]
    _assert_matches_server(dest_dir, api.videos)


def test_resume_after_async_failed_partway_in_another_dataset(tmp_path):
    """A complete dataset next to one whose download was cut off, resumed by the sync path."""
    dest_dir = str(tmp_path / "project")
    api = _FakeApi([_video(1), _video(2)])
    _download(api, dest_dir, use_async=True)
    # A cut-off download in a second dataset leaves a partial file with no annotation.
    partial = tmp_path / "project" / "other" / "video" / "clip_9.mp4"
    partial.parent.mkdir(parents=True)
    (tmp_path / "project" / "other" / "ann").mkdir()
    partial.write_bytes(b"\0" * 10)

    api.downloaded.clear()
    _download(api, dest_dir, use_async=False, resume_download=True)

    assert api.downloaded == []
    _assert_matches_server(dest_dir, api.videos)
