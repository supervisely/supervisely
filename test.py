import os
import shutil
from urllib.parse import urljoin, urlparse

import jwt
import requests
from dotenv import load_dotenv
from tqdm import tqdm

import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, SelectDataset, Text

# for convenient debug, has no effect in production
# load_dotenv("local.env")
# load_dotenv(os.path.expanduser("~/supervisely.env"))

# api = sly.Api.from_env()

# filters = [
#     {"field": "updatedAt", "operator": "<", "value": "2023-12-15T12:38:33Z"},
#     # {"field": "groupId", "operator": "=", "value": "2"},
# ]
# projects = api.project.get_list_all(
#     page="all", filters=filters, sort="createdAt", sort_order="asc", per_page=2000
# )
# projects = api.project.get_list_all(
#     page=4, filters=filters, sort="createdAt", sort_order="asc", per_page=2000
# )
# projects = api.project.get_list_all(filters=filters, sort="createdAt", sort_order="asc")
# projects = api.project.get_archivation_list(to_day=5, skip_exported=True)
# print(projects)

# filter_1 = {"field": "updatedAt", "operator": "<", "value": "2023-12-03T14:53:00.952Z"}
# filter_2 = {"field": "updatedAt", "operator": ">", "value": "2023-04-03T14:53:00.952Z"}
# filters = [filter_1, filter_2]
# datasets = api.dataset.get_list_all(filters)
# print(datasets)

# api.project.remove_permanently(31694)

# responses = api.dataset.remove_permanently([82705, 82706], 1)
# print(1)
# api.dataset.remove_permanently([82397, 82495, 82565])

# # project_id = None
# # dataset_id = None

# button = Button("Random text")
# text = Text("press the button to get random text")
# # initialize widgets we will use in UI
# # select_dataset = SelectDataset(default_id=dataset_id, project_id=project_id, multiselect=False)
# # card = Card(
# #     title="Select Dataset",
# #     content=Container(widgets=[select_dataset]),
# # )
# layout = Container(widgets=[button, text])
# app = sly.Application(layout=layout)


# select_dataset.set_dataset_id([81516])
# TEAM_ID = 549
# WORKSPACE_ID = 1043
# size = sly.fs.get_file_size("Japan.mp4")
# progress = sly.tqdm.tqdm(desc="Uploading", total=size, unit_scale=True, unit="B")


# api.video.upload_path(81183, "Japan.mp4", "Japan.mp4", item_progress=progress)
# def _download_batch_with_retry(api: sly.Api, dataset_id, image_ids):
#     retry_cnt = 5
#     curr_retry = 1
#     try:
#         imgs_bytes = api.image.download_bytes(dataset_id, image_ids)
#         if len(imgs_bytes) != len(image_ids):
#             raise RuntimeError(
#                 f"Downloaded {len(imgs_bytes)} images, but {len(image_ids)} expected."
#             )
#         return imgs_bytes
#     except Exception as e:
#         # sly.logger.warn(f"Failed to download images... Error: {e}")
#         while curr_retry <= retry_cnt:
#             try:
#                 # sly.logger.warn(f"Retry {curr_retry}/{retry_cnt} to download images")
#                 time.sleep(2 * curr_retry)
#                 imgs_bytes = api.image.download_bytes(dataset_id, image_ids)
#                 if len(imgs_bytes) != len(image_ids):
#                     raise RuntimeError(
#                         f"Downloaded {len(imgs_bytes)} images, but {len(image_ids)} expected."
#                     )
#                 return imgs_bytes
#             except Exception as e:
#                 curr_retry += 1
#     raise RuntimeError(
#         f"Failed to download images with ids {image_ids}. Check your data and try again later."
#     )


# def download_project(
#     api: sly.Api,
#     project_id,
#     dest_dir,
#     dataset_ids=None,
#     log_progress=True,
#     batch_size=10,
#     save_image_meta=True,
# ):
#     dataset_ids = set(dataset_ids) if (dataset_ids is not None) else None
#     project_fs = sly.Project(dest_dir, sly.OpenMode.CREATE)
#     meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
#     project_fs.set_meta(meta)

#     for dataset_info in api.dataset.get_list(project_id):
#         dataset_id = dataset_info.id
#         if dataset_ids is not None and dataset_id not in dataset_ids:
#             continue

#         dataset_fs = project_fs.create_dataset(dataset_info.name)
#         images = api.image.get_list(dataset_id)

#         if save_image_meta:
#             meta_dir = os.path.join(dest_dir, dataset_info.name, "meta")
#             sly.fs.mkdir(meta_dir)
#             for image_info in images:
#                 meta_paths = os.path.join(meta_dir, image_info.name + ".json")
#                 sly.json.dump_json_file(image_info.meta, meta_paths)

#         ds_progress = None
#         if log_progress:
#             ds_progress = sly.Progress(
#                 "Downloading dataset: {!r}".format(dataset_info.name),
#                 total_cnt=len(images),
#             )

#         for batch in sly.batched(images, batch_size=batch_size):
#             image_ids = [image_info.id for image_info in batch]
#             image_names = [image_info.name for image_info in batch]

#             # download images
#             batch_imgs_bytes = _download_batch_with_retry(api, dataset_id, image_ids)

#             # download annotations in json format
#             ann_infos = api.annotation.download_batch(dataset_id, image_ids)
#             ann_jsons = [ann_info.annotation for ann_info in ann_infos]

#             for name, img_bytes, ann in zip(image_names, batch_imgs_bytes, ann_jsons):
#                 dataset_fs.add_item_raw_bytes(item_name=name, item_raw_bytes=img_bytes, ann=ann)

#             if log_progress:
#                 ds_progress.iters_done_report(len(batch))


# data_dir = "downloaded_project"

# download_dir = os.path.join(data_dir, f"{255527}_images")
# if os.path.exists(download_dir):
#     sly.fs.clean_dir(download_dir)
# download_project(
#     api,
#     255527,
#     download_dir,
#     log_progress=False,
#     batch_size=50,
#     save_image_meta=True,
# )


# def check_cyrillic_in_files(directory):
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             file_path = os.path.join(root, file)
#             try:
#                 with open(file_path, "r", encoding="utf-8") as f:
#                     lines = f.readlines()
#                     for line_num, line in enumerate(lines, start=1):
#                         if any("\u0400" <= c <= "\u04FF" for c in line):
#                             print(f"File: {file_path}, Line: {line_num}, Context: {line.strip()}")
#             except UnicodeDecodeError:
#                 pass


# directory_path = "./"
# check_cyrillic_in_files(directory_path)

# info = api.video.upload_paths(82389, ["file"], ["/home/ganpoweird/Work/supervisely/cars.mp4"])

# import mimetypes

# from supervisely._utils import generate_free_name


# def volumes_upload_paths(dataset_id, names, paths, progress_cb):
#     if progress_cb:
#         log_progress = True
#     else:
#         log_progress = False

#     parent_dirs = []
#     for path in paths:
#         parent_dirs.append(os.path.dirname(path))
#     parent_dirs = list(set(parent_dirs))

#     series_infos = {}
#     nrrd_paths = []
#     for parent_dir in parent_dirs:
#         series_infos.update(sly.volume.inspect_dicom_series(root_dir=parent_dir))
#         nrrd_paths.extend(sly.volume.inspect_nrrd_series(root_dir=parent_dir))

#     if len(series_infos) == 0 and len(nrrd_paths) == 0:
#         msg = "No DICOM volumes were found. Please, check your input directory."
#         description = f"Supported formats: {sly.volume.volume.ALLOWED_VOLUME_EXTENSIONS} (in archive or directory)."
#         sly.logger.warning(f"{msg} {description}")

#     else:
#         used_volumes_names = [volume.name for volume in api.volume.get_list(dataset_id)]

#         for serie_id, files in series_infos.items():
#             item_path = files[0]
#             if sly.volume.get_extension(path=item_path) is None:
#                 sly.logger.warning(
#                     f"Can not recognize file extension {item_path}, serie will be skipped"
#                 )
#                 continue
#             name = f"{serie_id}.nrrd"
#             name = generate_free_name(
#                 used_names=used_volumes_names, possible_name=name, with_ext=True
#             )
#             used_volumes_names.append(name)
#             api.volume.upload_dicom_serie_paths(
#                 dataset_id=dataset_id,
#                 name=name,
#                 paths=files,
#                 log_progress=log_progress,
#                 anonymize=True,
#             )

#         for nrrd_path in nrrd_paths:
#             name = sly.fs.get_file_name_with_ext(path=nrrd_path)
#             name = generate_free_name(
#                 used_names=used_volumes_names, possible_name=name, with_ext=True
#             )
#             used_volumes_names.append(name)
#             api.volume.upload_nrrd_serie_path(
#                 dataset_id=dataset_id, name=name, path=nrrd_path, log_progress=log_progress
#             )


# volumes_upload_paths(
#     82398, ["0015.DCM"], ["/home/ganpoweird/Work/supervisely/volumes/0015.DCM"], progress_cb=None
# )
# print(info)


# input_path = "/home/ganpoweird/Work/supervisely_cli"
# temp_dir = "/tmp/deb_temp/supervisely_cli"
# binary_dir = os.path.join(temp_dir, "usr/local/bin")
# os.makedirs(binary_dir, exist_ok=True)

# shutil.copy(input_path, os.path.join(binary_dir))


# def authenticate():
#     server_url = urljoin(server, "api/account")
#     headers = {"Content-Type": "application/json"}
#     payload = {"login": login, "password": password}

#     response = requests.post(server_url, data=payload)
#     if response.status_code == 200:
#         data = response.json()
#         jwt_token = data.get("token", None)
#         token_parts = jwt_token.split(".")
#         decoded_token = jwt.decode(jwt_token, options={"verify_signature": False})
#         api_token = decoded_token.get("apiToken")
#         print(api_token)


# api = sly.Api.from_credentials(server, login, password, is_overwrite=False)
# # LoginInfo = sly.api.LoginInfo(server, login, password).log_in()


# print("sd")
# if sly.is_development():
#     load_dotenv("debug.env")
#     load_dotenv(os.path.expanduser("~/supervisely_dev_free.env"))

# api = sly.Api.from_env()


# TEAM_ID = 549
# WORKSPACE_ID = 1043
# size = sly.fs.get_file_size("1.76 GB.tar")
# progress = tqdm(desc="Uploading", total=size, unit_scale=True, unit="B")

# api.file.upload(
#     TEAM_ID,
#     "1.76 GB.tar",
#     "/tmp/1.76 GB.tar",
#     progress_cb=progress,
# )

api = sly.Api.from_env()
dataset = 83186
rename_if_exists = True
names = ["image1.png", "image1_004.png"]
if rename_if_exists is True:
    new_names = []
    for name in names:
        new_name = api.image.get_free_name(dataset, name)

        if new_name != name:
            if new_name in names or new_name in new_names:
                file_name = sly.fs.get_file_name(new_name)
                file_ext = sly.fs.get_file_ext(new_name)
                file_name = file_name + "_deduplicated"
                new_name = file_name + file_ext
            new_names.append(new_name)
        else:
            new_names.append(name)
    names = new_names
print(names)
