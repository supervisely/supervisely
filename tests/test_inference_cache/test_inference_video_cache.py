import os
import numpy as np
import pytest
from pathlib import Path

from supervisely.nn.inference.cache import InferenceVideoCache


# Test methods
def test_persistent_cache_base_function(api_mock, app_mock, get_data_dir_mock, tmp_path: Path):
    base_folder_in_app_data = "smart_tool_cache"
    inf_cache = InferenceVideoCache(
        app_mock, 2, max_number_of_videos=2, base_folder_in_app_data=base_folder_in_app_data
    )
    video_ids = [1, 2, 3]
    frame_indexes = [[1, 2], 3, [4, 5]]

    for vid, frames in zip(video_ids, frame_indexes):
        inf_cache.add_frames_to_cache(api_mock, vid, frames)

        if isinstance(frames, int):
            frames = [frames]

        save_path = tmp_path / base_folder_in_app_data / str(vid)
        imgs = [f"{fi}.png" for fi in frames]

        assert save_path.exists()
        assert sorted(os.listdir(save_path)) == sorted(imgs)

    assert not (tmp_path / "1").exists()
