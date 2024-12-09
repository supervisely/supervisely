import os

from tqdm import tqdm

import supervisely as sly

save_path = "/home/ganpoweird/Work/supervisely/supervisely/test_imgs/"
api = sly.Api.from_env()
TEAM_ID = 451  # TODO change to your team_id
WORKSPACE_ID = 712  # TODO change to your workspace_id
DATASET_ID = 98361  # TODO change to your dataset_id from which you want to download images


def download_images():

    sly.fs.ensure_base_path(save_path)
    imgs = api.image.get_list(DATASET_ID)
    ids = [img.id for img in imgs]
    progress = tqdm(total=len(ids), leave=False, desc="Downloading images")
    paths = [save_path + img.name for img in imgs]
    api.image.download_paths(DATASET_ID, ids, paths, progress_cb=progress)
    return paths


def upload_images(paths):
    project = api.project.create(
        WORKSPACE_ID, "MULTIPART_test_project", change_name_if_conflict=True
    )
    dataset = api.dataset.create(project.id, "MULTIPART_test_dataset", change_name_if_conflict=True)
    names = [os.path.basename(img) for img in paths]
    progress = tqdm(total=len(names), leave=False, desc="Uploading images to project")
    infos = api.image.upload_paths(dataset.id, names, paths, progress_cb=progress)
    if len(infos) != len(paths):
        raise RuntimeError("Not all images were uploaded to project")


def upload_files(paths):
    remote_paths = ["/test_multipart_monitor/" + os.path.basename(path) for path in paths]
    progress = tqdm(total=len(paths), leave=False, desc="Uploading images to Team Files")
    infos = api.file.upload_bulk(TEAM_ID, paths, remote_paths, progress_cb=progress)
    if len(infos) != len(paths):
        raise RuntimeError("Not all images were uploaded to Team Files")


if __name__ == "__main__":
    paths = download_images()  # MultipartDecoder
    upload_images(paths)  # MultipartEncoder
    upload_files(paths)  # MultipartEncoderMonitor
