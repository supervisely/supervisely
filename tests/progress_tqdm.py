from dotenv import load_dotenv
import os, math
from time import sleep

import supervisely as sly

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")
api = sly.Api()

# print("___________START_________")

# TF_PATH = "/my_project.tar"
TF_FILEPATH = "/cardio.tar"
LOC_FILEPATH = "/tmp/prog_test.tar"
TF_DIRPATH = "/19399_cardio/"
LOC_DIRPATH = "/tmp/_local/"

TEAM_ID = 449
PROJECT_ID = 18142

obj = api.file.list(TEAM_ID, TF_DIRPATH, recursive=True, return_type="fileinfo")[0]

# obj.fullStorageUrl
# obj.aaaaaaaaaaaaa

api.file.list2(TEAM_ID, TF_DIRPATH, recursive=True)

"""
Output:
[Dict_or_FileInfo(info_json) for info_json in response.json()] == response.json()
True
"""

batch_size = 10
data = range(100)

from tqdm import tqdm

# from supervisely.task.progress import WrapTqdm as tqdm

# os.environ["ENV"] = "production"

# with tqdm(total=len(data)) as pbar:
#     for batch in sly.batched(data, batch_size):
#         for item in batch:
#             sleep(0.1)
#         pbar.update(batch_size)

import shutil

# p = tqdm(
#     desc="api.file.download tqdm dev",
#     total=api.file.get_directory_size(TEAM_ID, TF_FILEPATH),
#     # unit="B",
#     # unit_scale=True,
#     is_size=True,
# )
# api.file.download(TEAM_ID, TF_FILEPATH, LOC_FILEPATH, progress_cb=p)
# os.remove(LOC_FILEPATH)


# n_count = api.project.get_info_by_id(17732).items_count
# p = get_p_for_test("sly.download", "it", "dev", n_count)
# sly.download(api, 17732, LOC_DIRPATH, progress_cb=p)
# shutil.rmtree(LOC_DIRPATH)
# for method, project_id in zip(
# [
# sly.download,
# sly.download_pointcloud_episode_project,
# sly.download_pointcloud_project,
# sly.download_project,
# sly.download_video_project,
# ],
#     [
#         # 17732,
#         # 18593,
#         18592,
#         # 17732,
#         # 18144,
#     ],
# ):
#     n_count = api.project.get_info_by_id(project_id).items_count
#     p = get_p_for_test("download", "it", "dev", n_count)
#     method(api, project_id, LOC_DIRPATH, progress_cb=p)
#     shutil.rmtree(LOC_DIRPATH)

# n_count = api.project.get_info_by_id(18594).items_count
# # p = get_p_for_test("download", "it", "dev", n_count)

# p = tqdm(
#     desc="sly.download_volume_project tqdm dev",
#     total=n_count,
# )

# sly.download(api, 18594, LOC_DIRPATH, progress_cb=p, download_volumes=False)

# # sly.download_volume_project(api, 18594, LOC_DIRPATH, progress_cb=p, download_volumes=False)
# shutil.rmtree(LOC_DIRPATH)


# files = []
# for r, d, fs in os.walk(LOC_DIRPATH):
#     files.extend(os.path.join(r, file) for file in fs)
# n_count = len(files) - 1  # minus meta.json
# p = tqdm(
#     desc="sly.upload_project",
#     total=n_count,
# )
# sly.upload_project(LOC_DIRPATH, api, 691, progress_cb=p)

# p = get_p_for_test("api.annotation.download_batch", "it", "prod", 2)
# api.annotation.download_batch(59589, [19425323, 19425319], progress_cb=p)

# p = get_p_for_test("api.annotation.download_batch", "it", "prod", 2)
# api.annotation.download_json_batch(59589, [19425323, 19425319], progress_cb=p)

# n_count = api.dataset.get_info_by_id(59589).items_count
# p = get_p_for_test("api.annotation.download_batch", "it", "dev", n_count)
# api.annotation.get_list(dataset_id=59589, progress_cb=p)

# api.annotation.upload_anns  # progress_cb(len(batch))
# api.annotation.upload_jsons  # progress_cb(len(batch))
# api.annotation.upload_paths  # progress_cb(len(batch))

# p(len(iteration))
# api.agent.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.agent.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))

# api.app.upload_dtl_archive # progress_cb(read_mb)
# api.app.upload_files # Method is unavailable

# api.dataset.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.dataset.get_list_all_pages_generator # progress_cb(len(results)), progress_cb(len(results))

# from supervisely.cli.teamfiles.teamfiles_download import download_directory_run

# shutil.rmtree("/tmp/test-dir")

# download_directory_run(449, "/my-training/", "/tmp/test-dir", filter=".tfevents.", ignore_if_not_exists=True)

# shutil.rmtree(LOC_DIRPATH)
# p = tqdm(
#     desc=f"download_directory1",
#     total=api.file.get_directory_size(TEAM_ID, TF_DIRPATH),
#     unit="B",
#     unit_scale=True,
# )
# api.file.download_directory(TEAM_ID, TF_DIRPATH, LOC_DIRPATH, progress_cb=p)
# shutil.rmtree(LOC_DIRPATH)

# p = tqdm(
#     desc=f"download_directory2",
#     total=api.file.get_directory_size(TEAM_ID, TF_DIRPATH),
#     unit="B",
#     unit_scale=True,
# )
# api.file.download_directory(TEAM_ID, TF_DIRPATH, LOC_DIRPATH, progress_cb=p)

# # shutil.rmtree(LOC_DIRPATH)/

# p = tqdm(
#     desc=f"upload_directory",
#     total=sly.fs.get_directory_size(LOC_DIRPATH),
#     unit="B",
#     unit_scale=True,
# )
# api.file.upload_directory(
#     TEAM_ID,
#     LOC_DIRPATH,
#     TF_DIRPATH,
#     progress_size_cb=p,
# )

# p = tqdm(
#     desc=f"download",
#     total=api.file.get_directory_size(TEAM_ID, TF_FILEPATH),
#     unit="B",
#     unit_scale=True,
# )
# api.file.download(TEAM_ID, TF_FILEPATH, LOC_FILEPATH, progress_cb=p)

# p = tqdm(
#     desc=f"upload",
#     total=sly.fs.get_file_size(LOC_FILEPATH),
#     unit="B",
#     unit_scale=True,
# )
# api.file.upload(
#     TEAM_ID,
#     LOC_FILEPATH,
#     TF_FILEPATH,
#     progress_cb=p,
# )

# api.github.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.github.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))

# p = get_p_for_test("api.image.download_bytes", "B", "dev", 3)
# api.image.download_bytes(61226, [19489032, 19489521, 19489360], progress_cb=p)

# api.image.download_nps  # progress_cb(1)

# local_save_dir = "vid/"
# save_paths = []
# image_ids = [19489032, 19489521, 19489360]
# img_infos = api.image.get_info_by_id_batch(image_ids)


# # os.environ["ENV"] = "production"
# p = tqdm(desc="Images downloaded: ", total=len(img_infos))
# # p = tqdm(None, "Images downloaded: ", len(img_infos))
# for img_info in img_infos:
#     save_paths.append(os.path.join(local_save_dir, img_info.name))

# api.image.download_paths(61226, image_ids, save_paths, progress_cb=p)

# api.image.download_paths  # progress_cb(1)
# api.image.download_paths_by_hashes  # progress_cb(1)
# api.image.get_info_by_id_batch  # progress_cb(len(batch))
# api.image.get_list_all_pages  # progress_cb(len(results))
# api.image.get_list_all_pages_generator  # progress_cb(len(results))
# api.image.add_tag_batch  # progress_cb(len(batch_ids))
# api.image.check_existing_hashes # progress_cb(len(hashes_batch))


# ds = 61387
# ds_lemon_img_infos = api.image.get_list(ds)
# fruit_img_ids = []
# for lemon_img_info in ds_lemon_img_infos:
#     fruit_img_ids.append(lemon_img_info.id)
# n_count = api.dataset.get_info_by_id(ds).items_count
# p = get_p_for_test("api.image.download_bytes", "it", "prod", n_count)
# dst_ds = 61389
# ds_fruit_img_infos = api.image.copy_batch(dst_ds, fruit_img_ids, progress_cb=p)
# api.image.copy_batch  # progress_cb(len(images)) #TODODONE


# api.image.copy_batch_optimized #progress_cb(len(images))
# api.image.move_batch  # progress_cb(len(images))
# api.image.move_batch_optimized # progress_cb(len(images))
# api.image.remove_batch #progress_cb(len(ids_batch))
# api.image.upload_hashes  # progress_cb(len(images))
# api.image.upload_ids # progress_cb(len(images)),
# api.image.upload_nps # progress_cb(len(remote_hashes)), progress_cb(len(hashes_rcv))
# api.image.upload_paths # progress_cb(len(remote_hashes)), progress_cb(len(hashes_rcv))
# api.image.get_info_by_id_batch # progress_cb(len(batch))

# api.img_ann_tool.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.img_ann_tool.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))

# api.import_storage.get_list_all_pages # progress_cb(len(results)) progress_cb(len(temp_items))
# api.import_storage.get_list_all_pages_generator # progress_cb(len(results)) progress_cb(len(results))

# api.labeling_job.get_list_all_pages  # progress_cb(len(results)) progress_cb(len(temp_items))
# api.labeling_job.get_list_all_pages_generator  # progress_cb(len(results)) progress_cb(len(results))
api.labeling_job.get_activity  # api.team.get_activity #TODODONE

# api.model.download_to_dir  # progress_cb(read_mb)
# api.model.download_to_tar  # progress_cb(read_mb)
# api.model.get_list_all_pages  # progress_cb(len(results)) progress_cb(len(temp_items))
# api.model.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))
# api.model.remove_batch  # progress_cb(1)
# api.model.upload  # progress_cb(read_mb)

# api.object_class.get_list_all_pages  # progress_cb(len(results)) progress_cb(len(temp_items))
# api.object_class.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))

# api.plugin.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.plugin.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))

# api.pointcloud.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.pointcloud.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))
# api.pointcloud.remove_batch  # progress_cb(len(ids_batch))
# api.pointcloud.upload_paths  # progress_cb(len(remote_hashes)), progress_cb(len(batch))
# api.pointcloud.upload_related_images  # progress_cb(len(remote_hashes)), progress_cb(len(batch))
# api.pointcloud.upload_hashes  # progress_cb(len(images))

# api.pointcloud_episode.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.pointcloud_episode.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))
# api.pointcloud_episode.remove_batch  # progress_cb(len(ids_batch))
# api.pointcloud_episode.upload_hashes  # progress_cb(len(images))
# api.pointcloud_episode.upload_related_images  # progress_cb(len(remote_hashes)) progress_cb(len(batch))
# api.pointcloud_episode.upload_paths  # progress_cb(len(remote_hashes)) progress_cb(len(batch))

# api.project.download_images_tags  # progress_cb(1)
# p = get_p_for_test("api.team.get_activity", "it", "prod", 0)
# api.project.get_activity  # team get_activity # TODODONE
# api.project.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))
# api.project.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.project.remove_batch  # progress_cb(1)

# api.remote_storage.download_path  # progress_cb(len(chunk))
# api.remote_storage.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.remote_storage.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))

# api.report.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.report.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))

# api.role.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.role.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))

# api.task.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.task.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))
# api.task.upload_dtl_archive  # progress_cb(read_mb)
# api.task.upload_files  # progress_cb(len(remote_hashes)), progress_cb(len(content_dict))

# from supervisely.api.team_api import ActivityAction as aa

# labeling_actions = [
#     aa.ATTACH_TAG,
#     aa.UPDATE_TAG_VALUE,
#     aa.DETACH_TAG,
#     aa.ADD_MEMBER,
# ]
# os.environ["ENV"] = "production"
# p = tqdm(desc=f"get_activity", total=0, is_size=True)

# # p = sly.Progress(message="api.team.get_activity", total_cnt=0)
# sfijgf = api.team.get_activity(449, filter_actions=labeling_actions, progress_cb=p)


# api.team.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.team.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))

# api.user.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.user.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))

# TODO item_progress discusssion
# api.video.download_path  # progress_cb(len(chunk))
# api.video.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.video.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))
# api.video.remove_batch  # progress_cb(len(ids_batch))
# api.video.upload_hashes  # progress_cb(len(images))
# dataset_id = 60565
# video_names = [
#     "7777.mp4",
# ]
# video_paths = [
#     "7777.mp4",
# ]

# p = tqdm(total=1)
# # p = sly.Progress(total_cnt=1, message="Upl")
# video_infos = api.video.upload_paths(
#     dataset_id=dataset_id, names=video_names, paths=video_paths, progress_cb=p  # .iters_done_report
# )
# api.video.upload_paths()  # progress_cb(len(remote_hashes)), progress_cb(len(hashes_rcv))

# src_dataset_id = 61229
# info = api.dataset.create(20697, "tst", change_name_if_conflict=True)
# dst_dataset_id = info.id

# hashes = []
# names = []
# metas = []
# volume_infos = api.volume.get_list(src_dataset_id)

# # Create lists of hashes, volumes names and meta information for each volume
# for volume_info in volume_infos:
#     hashes.append(volume_info.hash)
#     # It is necessary to upload volumes with the same names(extentions) as in src dataset
#     names.append(volume_info.name)
#     metas.append(volume_info.meta)

# p = tqdm(desc="api.volume.upload_hashes", total=len(hashes))
# new_volumes_info = api.volume.upload_hashes(
#     dataset_id=dst_dataset_id,
#     names=names,
#     hashes=hashes,
#     progress_cb=p,
#     metas=metas,
# )


# api.volume.upload_hashes  # progress_cb(len(volumes))

# src_dataset_id = 61229
# volume_infos = api.volume.get_list(src_dataset_id)
# volume_id = volume_infos[0].id
# volume_info = api.volume.get_info_by_id(id=volume_id)

# download_dir_name = "vid/"
# path = os.path.join(download_dir_name, volume_info.name)
# if os.path.exists(path):
#     os.remove(path)

# # os.environ["ENV"] = "production"
# p = tqdm(desc="Volumes upload: ", total=volume_info.sizeb, is_size=True)
# api.volume.download_path(volume_info.id, path, progress_cb=p)

# if os.path.exists(path):
#     print(f"Volume (ID {volume_info.id}) successfully downloaded.")
# api.volume.download_path  # progress_cb(len(chunk))
# api.volume.get_list_all_pages  ## progress_cb(len(results)), progress_cb(len(temp_items))
# api.volume.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))
# api.volume.remove_batch  # progress_cb(len(ids_batch))
# api.volume.upload_np  # progress_cb(1), progress_cb(len(batch))
# api.volume.upload_nrrd_series_paths  # progress_cb(1)

# api.workspace.get_list_all_pages  # progress_cb(len(results)), progress_cb(len(temp_items))
# api.workspace.get_list_all_pages_generator  # progress_cb(len(results)), progress_cb(len(results))


# /home/grokhi/supervisely/sdk/supervisely/supervisely/io/fs_cache.py
# write_objects

# /home/grokhi/supervisely/sdk/supervisely/supervisely/io/fs.py

# /home/grokhi/supervisely/sdk/supervisely/supervisely/project/pointcloud_episode_project.py
# /home/grokhi/supervisely/sdk/supervisely/supervisely/project/pointcloud_project.py

# _upload_uniq_videos_single_req
