import SimpleITK as sitk

from supervisely.api.module_api import ApiField, RemoveableBulkModuleApi
from supervisely.io.fs import (
    get_file_ext,
    get_file_name,
    get_bytes_hash,
)
from supervisely import volume
import supervisely.volume.nrrd_encoder as nrrd_encoder
from supervisely._utils import batched
from supervisely import logger
from supervisely.task.progress import Progress


class VolumeApi(RemoveableBulkModuleApi):
    @staticmethod
    def info_sequence():
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.LINK,
            ApiField.HASH,
            ApiField.MIME,
            ApiField.EXT,
            ApiField.SIZE,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.META,
            ApiField.PATH_ORIGINAL,
            ApiField.FULL_STORAGE_URL,
            ApiField.TAGS,
            ApiField.TEAM_ID,
            ApiField.WORKSPACE_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.FILE_META,
            ApiField.FIGURES_COUNT,
            ApiField.ANN_OBJECTS_COUNT,
            ApiField.PROCESSING_PATH,
        ]

    @staticmethod
    def info_tuple_name():
        return "VolumeInfo"

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        return res

    def get_list(self, dataset_id, filters=None, sort="id", sort_order="asc"):
        return self.get_list_all_pages(
            "volumes.list",
            {
                ApiField.DATASET_ID: dataset_id,
                ApiField.FILTER: filters or [],
                ApiField.SORT: sort,
                ApiField.SORT_ORDER: sort_order,
            },
        )

    def upload_hash(self, dataset_id, name, hash, meta=None):
        metas = None if meta is None else [meta]
        return self.upload_hashes(dataset_id, [name], [hash], metas=metas)[0]

    def upload_hashes(self, dataset_id, names, hashes, progress_cb=None, metas=None):
        return self._upload_bulk_add(
            lambda item: (ApiField.HASH, item),
            dataset_id,
            names,
            hashes,
            progress_cb,
            metas=metas,
        )

    def _upload_bulk_add(
        self, func_item_to_kv, dataset_id, names, items, progress_cb=None, metas=None
    ):
        results = []

        if len(names) == 0:
            return results
        if len(names) != len(items):
            raise RuntimeError(
                'Can not match "names" and "items" lists, len(names) != len(items)'
            )

        if metas is None:
            metas = [{}] * len(names)
        else:
            if len(names) != len(metas):
                raise RuntimeError(
                    'Can not match "names" and "metas" len(names) != len(metas)'
                )

        for batch in batched(list(zip(names, items, metas))):
            volumes = []
            for name, item, meta in batch:
                item_tuple = func_item_to_kv(item)
                image_data = {ApiField.NAME: name, item_tuple[0]: item_tuple[1]}
                if len(meta) != 0 and type(meta) == dict:
                    image_data[ApiField.META] = meta
                volumes.append(image_data)

            response = self._api.post(
                "volumes.bulk.add",
                {ApiField.DATASET_ID: dataset_id, ApiField.VOLUMES: volumes},
            )
            if progress_cb is not None:
                progress_cb(len(volumes))

            for info_json in response.json():
                results.append(self._convert_json_info(info_json))
        return results

    def upload_np(self, dataset_id, name, np_data, meta, progress_cb=None):
        ext = get_file_ext(name)
        if ext != ".nrrd":
            raise ValueError(
                "Name has to be with .nrrd extension, for example: my_volume.nrrd"
            )

        volume_bytes = volume.encode(np_data, meta)
        volume_hash = get_bytes_hash(volume_bytes)
        self._api.image._upload_data_bulk(lambda v: v, [(volume_bytes, volume_hash)])
        volume_info = self.upload_hash(
            dataset_id, f"{get_file_name(name)}.nrrd", volume_hash, meta
        )
        progress_cb(1)  # upload volume

        # slice all directions
        # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/03_Image_Details.html#Conversion-between-numpy-and-SimpleITK
        # x = 1 - sagittal
        # y = 1 - coronal
        # z = 1 - axial

        for (plane, dimension) in zip(["sagittal", "coronal", "axial"], np_data.shape):
            slices = []
            for i in range(dimension):
                try:
                    normal = {"x": 0, "y": 0, "z": 0}

                    if plane == "sagittal":
                        pixel_data = np_data[i, :, :]
                        normal["x"] = 1
                    elif plane == "coronal":
                        pixel_data = np_data[:, i, :]
                        normal["y"] = 1
                    elif plane == "axial":
                        pixel_data = np_data[:, :, i]
                        normal["z"] = 1
                    else:
                        raise ValueError(f"Unknown plane {plane}")

                    img_bytes = nrrd_encoder.encode(
                        pixel_data, header={"encoding": "gzip"}, compression_level=1
                    )

                    img_hash = get_bytes_hash(img_bytes)
                    self._api.image._upload_data_bulk(
                        lambda v: v, [(img_bytes, img_hash)]
                    )

                    slices.append(
                        {
                            "hash": img_hash,
                            "sliceIndex": i,
                            "normal": normal,
                        }
                    )

                    if len(slices) > 50:
                        self._upload_slices_bulk(volume_info.id, slices, progress_cb)
                        slices.clear()

                except Exception as e:
                    exc_str = str(e)
                    logger.warn(
                        "File skipped due to error: {}".format(exc_str),
                        exc_info=True,
                        extra={
                            "exc_str": exc_str,
                            "file_path": name,
                        },
                    )

            if len(slices) > 0:
                self._upload_slices_bulk(volume_info.id, slices, progress_cb)
                slices.clear()

        return volume_info

    def upload_dicom_serie_paths(
        self,
        dataset_id,
        name,
        paths,
        log_progress=True,
    ):
        volume_np, volume_meta = volume.read_serie_volume_np(paths)
        progress_cb = None
        if log_progress is True:
            progress_cb = Progress(
                f"Upload volume {name}", sum(volume_np.shape)
            ).iters_done_report
        return self.upload_np(dataset_id, name, volume_np, volume_meta, progress_cb)

    def upload_nrrd_path(path):
        raise NotImplementedError()

    def _upload_slices_bulk(self, volume_id, items, progress_cb=None):
        results = []
        if len(items) == 0:
            return results

        for batch in batched(items):
            response = self._api.post(
                "volumes.slices.bulk.add",
                {ApiField.VOLUME_ID: volume_id, ApiField.VOLUME_SLICES: batch},
            )
            if progress_cb is not None:
                progress_cb(len(batch))
            results.extend(response.json())
        return results
