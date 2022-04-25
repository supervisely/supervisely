# coding: utf-8

import os
import numpy as np

from supervisely.io.json import load_json_file, dump_json_file
from supervisely import TaskPaths, ProjectType
from supervisely._utils import get_bytes_hash
import supervisely as sly

from supervisely.volume.parsers import dicom, nrrd, nifti
from supervisely.volume.nrrd_encoder import encoder as nrrd_encoder


def create_project(api, workspace_id, project_name, append_to_existing_project):
    if append_to_existing_project is True:
        dst_project = api.project.get_info_by_name(workspace_id, project_name)
        if dst_project is None:
            raise RuntimeError("Project {!r} not found".format(project_name))
    else:
        dst_project = api.project.create(
            workspace_id,
            project_name,
            type=ProjectType.VOLUMES,
            change_name_if_conflict=True,
        )
    return dst_project


def rescale_slope_intercept(value, slope, intercept):
    return value * slope + intercept


def normalize_volume_meta(meta):
    meta["intensity"]["min"] = rescale_slope_intercept(
        meta["intensity"]["min"],
        meta["rescaleSlope"],
        meta["rescaleIntercept"],
    )

    meta["intensity"]["max"] = rescale_slope_intercept(
        meta["intensity"]["max"],
        meta["rescaleSlope"],
        meta["rescaleIntercept"],
    )

    if "windowWidth" not in meta:
        meta["windowWidth"] = meta["intensity"]["max"] - meta["intensity"]["min"]

    if "windowCenter" not in meta:
        meta["windowCenter"] = meta["intensity"]["min"] + meta["windowWidth"] / 2

    return meta


def process_file(api, project_info, dataset_info, entry_path, parser_type):
    def_volume_name = os.path.relpath(
        os.path.splitext(entry_path)[0], TaskPaths.DATA_DIR
    ).replace(os.sep, "__")

    volumes = []
    images_cnt = 0

    if parser_type == "dicom":
        volumes = dicom.load(entry_path)
    elif parser_type == "nrrd":
        volumes = nrrd.load(entry_path)
    elif parser_type == "nifti":
        volumes = nifti.load(entry_path)

    for (volume_name, volume, volume_info) in volumes:
        if volume_name is None:
            volume_name = def_volume_name

        volume.system = "RAS"

        data = volume.aligned_data
        min_max = [int(np.amin(data)), int(np.amax(data))]

        volume_meta = normalize_volume_meta(
            {
                "channelsCount": 1,
                "rescaleSlope": 1,
                "rescaleIntercept": 0,
                **volume_info,
                "intensity": {
                    "min": min_max[0],
                    "max": min_max[1],
                },
                "dimensionsIJK": {
                    "x": data.shape[0],
                    "y": data.shape[1],
                    "z": data.shape[2],
                },
                "ACS": volume.system,
                "IJK2WorldMatrix": volume.aligned_transformation.flatten().tolist(),
            }
        )

        # to save normalized volume
        # from src.loaders import nrrd as nrrd_loader
        # nrrd_loader.save_volume('/sly_task_data/volume.nrrd', volume, src_order=False, src_system=False)

        progress = sly.Progress("Import volume: {}".format(volume_name), 1)

        norm_volume_bytes = nrrd_encoder.encode(
            data,
            header={
                "encoding": "gzip",
                "space": volume.system.upper(),
                "space directions": volume.aligned_transformation[:3, :3].T.tolist(),
                "space origin": volume.aligned_transformation[:3, 3].tolist(),
            },
            compression_level=1,
        )

        norm_volume_hash = get_bytes_hash(norm_volume_bytes)
        api.image._upload_data_bulk(
            lambda v: v, [(norm_volume_bytes, norm_volume_hash)]
        )

        [volume_result] = api.image.upload_volume(
            {
                "datasetId": dataset_info.id,
                "volumes": [
                    {
                        "hash": norm_volume_hash,
                        "name": f"{volume_name}.nrrd",
                        "meta": volume_meta,
                    },
                ],
            }
        )

        progress.iter_done_report()

        progress = sly.Progress(
            "Import volume slices: {}".format(volume_name), sum(data.shape)
        )

        for (plane, dimension) in zip(["sagittal", "coronal", "axial"], data.shape):
            for i in range(dimension):
                try:
                    normal = {"x": 0, "y": 0, "z": 0}

                    if plane == "sagittal":
                        pixel_data = data[i, :, :]
                        normal["x"] = 1
                    elif plane == "coronal":
                        pixel_data = data[:, i, :]
                        normal["y"] = 1
                    else:
                        pixel_data = data[:, :, i]
                        normal["z"] = 1

                    img_bytes = nrrd_encoder.encode(
                        pixel_data, header={"encoding": "gzip"}, compression_level=1
                    )

                    img_hash = get_bytes_hash(img_bytes)
                    api.image._upload_data_bulk(lambda v: v, [(img_bytes, img_hash)])

                    cur_img = {
                        "hash": img_hash,
                        "sliceIndex": i,
                        "normal": normal,
                    }

                    api.image._upload_volume_slices_bulk_add_dict(
                        volume_result["id"], [cur_img], None
                    )

                    images_cnt += 1

                except Exception as e:
                    exc_str = str(e)
                    sly.logger.warn(
                        "File skipped due to error: {}".format(exc_str),
                        exc_info=True,
                        extra={
                            "exc_str": exc_str,
                            "file_path": entry_path,
                        },
                    )

                progress.iter_done_report()

    return images_cnt


def main():
    task_config = load_json_file(TaskPaths.TASK_CONFIG_PATH)

    server_address = task_config["server_address"]
    token = task_config["api_token"]
    append_to_existing_project = task_config["append_to_existing_project"]

    api = sly.Api(server_address, token)
    task_info = api.task.get_info_by_id(task_config["task_id"])
    # @TODO: migrate to passing workspace id via the task config.
    project_info = create_project(
        api,
        task_info["workspaceId"],
        task_config["res_names"]["project"],
        append_to_existing_project,
    )
    dataset_info = api.dataset.create(
        project_info.id, "ds0", change_name_if_conflict=True
    )

    total_counter = 0

    for root, dirs, files in os.walk(TaskPaths.DATA_DIR):
        dir_has_dcm_files = False

        for entry_name in files:
            full_entry_name = os.path.join(root, entry_name)
            parser_type = None

            entry_name_low = entry_name.lower()

            if entry_name_low.endswith("nrrd") or entry_name_low.endswith("nrrd.gz"):
                parser_type = "nrrd"
            elif entry_name_low.endswith("nii") or entry_name_low.endswith("nii.gz"):
                parser_type = "nifti"
            elif entry_name_low.endswith("dcm"):
                dir_has_dcm_files = True

            if parser_type is not None:
                total_counter += process_file(
                    api, project_info, dataset_info, full_entry_name, parser_type
                )

        if dir_has_dcm_files:
            total_counter += process_file(
                api, project_info, dataset_info, root, "dicom"
            )

    if total_counter == 0:
        raise RuntimeError("Result project is empty! No valid files found")

    dump_json_file(
        {"project_id": project_info.id},
        os.path.join(TaskPaths.RESULTS_DIR, "project_info.json"),
    )


if __name__ == "__main__":
    sly.main_wrapper("VOLUMES_IMPORT", main)
