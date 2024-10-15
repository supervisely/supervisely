import asyncio
from time import sleep, time

from tqdm import tqdm

import supervisely as sly

api = sly.Api.from_env()

# api.logger.setLevel("DEBUG")

full_path = "/videos/MP4_HEVC.mp4"
team_id = 567
local_path = "/home/ganpoweird/Work/supervisely/video/video.mp4"
files = (
    ("12391768_3840_2160_30fps.mp4", "/videos/12391768_3840_2160_30fps.mp4"),
    # ("1fps.mp4", "/videos/1fps.mp4"),
    ("MOV_h264.mov", "/videos/MOV_h264.mov"),
    ("MP4_HEVC.mp4", "/videos/MP4_HEVC.mp4"),
)


async def download_files(api: sly.Api, team_id, files):
    sema = asyncio.Semaphore(10)
    save_path = "/home/ganpoweird/Work/supervisely/video/"
    tasks = []
    for name, path in files:
        task = api.file.async_download(
            team_id, path, save_path + name, sema, show_file_progress=True
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


# api.file.download_directory(team_id, "/videos/", "/home/ganpoweird/Work/supervisely/video")


# def run_async():
#     start = time()
#     asyncio.run(download_files(api, team_id, files))
#     finish = time() - start
#     return finish


asyncio.run(download_files(api, team_id, files))

# for _ in range(5):
#     for name, path in files:
#         api.file.download(team_id, path, "/home/ganpoweird/Work/supervisely/video/" + name)

# # time_list = []
# # for _ in range(5):
# #     sleep(2)
# #     time_list.append(run_async())

# # print(f"Min time: {round(min(time_list), 2)}")
# # print(f"Max time: {round(max(time_list), 2)}")
# # print(f"Average time: {round(sum(time_list) / len(time_list), 2)}")


# progress = tqdm(desc="Downloading", total=api.storage.get_info_by_path(team_id, full_path).sizeb, unit="B", unit_scale=True)
# with open(local_path, "wb") as f:
#     for chunk in api.stream(
#         "file-storage.download",
#         "POST",
#         {"teamId": team_id, "path": full_path},
#     ):
#         f.write(chunk)
#         progress(len(chunk))
