import os

import numpy as np
from dotenv import load_dotenv

import supervisely as sly

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

dataset_id = 60894
image_ids = [19480707, 19480703, 19480706, 19480705, 19480704]
image_infos = [api.image.get_info_by_id(img_id) for img_id in image_ids]
image_hashes = [info.hash for info in image_infos]

image_nps = [api.image.download_np(img_id) for img_id in image_ids]

# with generator using id
image_nps_gen = [img for _, img in api.image.download_nps_generator(dataset_id, image_ids)]

# with generator using hash
image_nps_hashes = api.image.download_nps_by_hashes(image_hashes)

# check images loaders
for single, with_hash, with_id in zip(image_nps, image_nps_hashes, image_nps_gen):
    assert np.allclose(single, with_hash)
    assert np.allclose(single, with_id)


video_id = 20407355
frame_indexes = list(range(5))

frames_bulk = api.video.frame.download_nps(video_id, frame_indexes)
frames_gen = [frame for _, frame in api.video.frame.download_nps_generator(video_id, frame_indexes)]

for bulk, gen in zip(frames_bulk, frames_gen):
    assert np.allclose(bulk, gen)
