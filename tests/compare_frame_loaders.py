"Compare InferenceImageCache and InferenceVideoInterface for video frames loading"
import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely import TinyTimer
from supervisely.nn.inference.cache import InferenceImageCache
from supervisely.nn.inference.video_inference import InferenceVideoInterface
from supervisely.sly_logger import logger

load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

video_id = 30406048
video_info = api.video.get_info_by_id(video_id)


def inf_video_interface(batch_size):
    video_interface = InferenceVideoInterface(
        api=api,
        start_frame_index=0,
        frames_count=video_info.frames_count,
        frames_direction="forward",
        video_info=video_info,
        imgs_dir="temp_data/video_inference",
        preparing_progress={},
    )
    tm = TinyTimer()
    i = 0
    video_interface.download_frames()
    logger.debug("download video time %s", TinyTimer.get_sec(tm))
    for batch in sly.batched(video_interface.images_paths, batch_size=batch_size):
        tm_i = TinyTimer()
        for img_path in batch:
            np = sly.image.read(path=img_path)
            i += 1
        delta_i = TinyTimer.get_sec(tm_i)
        print(
            f"{i}/{video_info.frames_count}, iter: {delta_i:.3f}, total: {TinyTimer.get_sec(tm):.3f} sec, {'*'*int(delta_i/0.1)}"
        )
    return TinyTimer.get_sec(tm), i


def inf_image_cache(batch_size):
    cache = InferenceImageCache(
        maxsize=sly.env.smart_cache_size(),
        ttl=60 * 1000,
        is_persistent=True,
        base_folder="temp_data/inference_cache",
    )

    tm = TinyTimer()
    cache.download_video(api=api, video_id=video_id, return_images=False)
    logger.debug("download video time %s", TinyTimer.get_sec(tm))
    print(video_id in cache._cache)
    i = 0
    for batch in sly.batched(range(video_info.frames_count), batch_size=batch_size):
        tm_i = TinyTimer()
        for np in cache.download_frames(api=api, video_id=video_id, frame_indexes=batch):
            i += 1
            continue
        delta_i = TinyTimer.get_sec(tm_i)
        print(
            f"{i}/{video_info.frames_count}, iter: {delta_i:.3f}, total: {TinyTimer.get_sec(tm):.3f} sec, {'*'*int(delta_i/0.1)}"
        )
    return TinyTimer.get_sec(tm), i


def compare():
    batch_size = 16
    cache_time, cache_count = inf_image_cache(batch_size)
    print(
        f"InferenceImageCache: {cache_time:.3f} sec, {cache_count} images, avg: {cache_time / cache_count:.6f} sec/image"
    )
    inter_time, inter_count = inf_video_interface(batch_size)
    print(
        f"InferenceVideoInterface: {inter_time:.3f} sec, {inter_count} images, avg: {inter_time / inter_count:.6f} sec/image"
    )


if __name__ == "__main__":
    compare()
