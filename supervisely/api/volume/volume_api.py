from typing import NamedTuple
from supervisely.api.module_api import ApiField, RemoveableBulkModuleApi
from supervisely.api.volume.volume_annotation_api import VolumeAnnotationAPI
from supervisely.api.volume.volume_object_api import VolumeObjectApi
from supervisely.api.volume.volume_figure_api import VolumeFigureApi

# from supervisely.api.volume.video_frame_api import VolumeFrameAPI
from supervisely.api.volume.volume_tag_api import VolumeTagApi

from supervisely.io.fs import ensure_base_path

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
from supervisely.imaging.image import read_bytes
from supervisely.volume_annotation.plane import Plane
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class VolumeInfo(NamedTuple):
    id: int
    name: str
    link: str
    hash: str
    mime: str
    ext: str
    sizeb: int
    created_at: str
    updated_at: str
    meta: dict
    path_original: str
    full_storage_url: str
    tags: list
    team_id: int
    workspace_id:  int
    project_id: int
    dataset_id: int
    file_meta: dict
    figures_count: int
    objects_count: int
    processing_path: str


class VolumeApi(RemoveableBulkModuleApi):
    def __init__(self, api):
        """
        :param api: Api class object
        """
        super().__init__(api)
        self.annotation = VolumeAnnotationAPI(api)
        self.object = VolumeObjectApi(api)
        # self.frame = VideoFrameAPI(api)
        self.figure = VolumeFigureApi(api)
        self.tag = VolumeTagApi(api)

    @staticmethod
    def info_sequence():
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.LINK,
            ApiField.HASH,
            ApiField.MIME,
            ApiField.EXT,
            ApiField.SIZEB3,
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
        return VolumeInfo(**res._asdict())

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

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, "volumes.info")

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

    def upload_np(
        self, dataset_id, name, np_data, meta, progress_cb=None, batch_size=30
    ):
        ext = get_file_ext(name)
        if ext != ".nrrd":
            raise ValueError(
                "Name has to be with .nrrd extension, for example: my_volume.nrrd"
            )
        from timeit import default_timer as timer

        logger.info("Start volume compression before upload...")
        start = timer()
        volume_bytes = volume.encode(np_data, meta)
        logger.info(f"Volume has been compressed in {timer() - start} seconds")

        logger.info(f"Start uploading bytes of 3d volume ...")
        start = timer()
        volume_hash = get_bytes_hash(volume_bytes)
        self._api.image._upload_data_bulk(lambda v: v, [(volume_bytes, volume_hash)])
        logger.info(
            f"3d Volume bytes has been sucessfully uploaded in {timer() - start} seconds"
        )
        volume_info = self.upload_hash(dataset_id, name, volume_hash, meta)
        if progress_cb is not None:
            progress_cb(1)  # upload volume

        # slice all directions
        # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/03_Image_Details.html#Conversion-between-numpy-and-SimpleITK
        # x = 1 - sagittal
        # y = 1 - coronal
        # z = 1 - axial
        planes = [Plane.SAGITTAL, Plane.CORONAL, Plane.AXIAL]

        for (plane, dimension) in zip(planes, np_data.shape):
            for batch in batched(list(range(dimension))):
                slices = []
                slices_bytes = []
                slices_hashes = []
                try:
                    for i in batch:
                        normal = Plane.get_normal(plane)

                        if plane == Plane.SAGITTAL:
                            pixel_data = np_data[i, :, :]
                        elif plane == Plane.CORONAL:
                            pixel_data = np_data[:, i, :]
                        elif plane == Plane.AXIAL:
                            pixel_data = np_data[:, :, i]
                        else:
                            raise ValueError(f"Unknown plane {plane}")

                        img_bytes = nrrd_encoder.encode(
                            pixel_data, header={"encoding": "gzip"}, compression_level=1
                        )

                        img_hash = get_bytes_hash(img_bytes)
                        slices_bytes.append(img_bytes)
                        slices_hashes.append(img_hash)
                        slices.append(
                            {
                                "hash": img_hash,
                                "sliceIndex": i,
                                "normal": normal,
                            }
                        )

                    if len(slices) > 0:
                        self._api.image._upload_data_bulk(
                            lambda v: v,
                            zip(slices_bytes, slices_hashes),
                        )
                        self._upload_slices_bulk(volume_info.id, slices, progress_cb)

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
        return volume_info

    def upload_dicom_serie_paths(self, dataset_id, name, paths, log_progress=True, anonymize=True):
        volume_np, volume_meta = volume.read_dicom_serie_volume_np(paths, anonymize=anonymize)
        progress_cb = None
        if log_progress is True:
            progress_cb = Progress(
                f"Upload volume {name}", sum(volume_np.shape)
            ).iters_done_report
        res = self.upload_np(dataset_id, name, volume_np, volume_meta, progress_cb)
        return self.get_info_by_name(dataset_id, name)

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

    def upload_nrrd_serie_path(self, dataset_id, name, path, log_progress=True):
        volume_np, volume_meta = volume.read_nrrd_serie_volume_np(path)
        progress_cb = None
        if log_progress is True:
            progress_cb = Progress(
                f"Upload volume {name}", sum(volume_np.shape)
            ).iters_done_report
        res = self.upload_np(dataset_id, name, volume_np, volume_meta, progress_cb)
        return self.get_info_by_name(dataset_id, name)

    def _download(self, id, is_stream=False):
        response = self._api.post(
            "volumes.download", {ApiField.ID: id}, stream=is_stream
        )
        return response

    def download_path(self, id, path, progress_cb=None):
        response = self._download(id, is_stream=True)
        ensure_base_path(path)

        with open(path, "wb") as fd:
            mb1 = 1024 * 1024
            for chunk in response.iter_content(chunk_size=mb1):
                fd.write(chunk)

                if progress_cb is not None:
                    progress_cb(len(chunk))

    def upload_nrrd_series_paths(
        self,
        dataset_id,
        names,
        paths,
        progress_cb=None,
    ):
        volume_infos = []
        for name, path in zip(names, paths):
            info = self.upload_nrrd_serie_path(
                dataset_id, name, path, log_progress=True
            )
            volume_infos.append(info)
            if progress_cb is not None:
                progress_cb(1)
        return volume_infos

    def download_slice_np(
        self,
        volume_id: int,
        slice_index: int,
        plane: Literal["sagittal", "coronal", "axial"],
        window_center: float = None,
        window_width: int = None,
    ):
        normal = Plane.get_normal(plane)
        meta = self.get_info_by_id(volume_id).meta

        if window_center is None:
            if "windowCenter" in meta:
                window_center = meta["windowCenter"]
            else:
                window_center = meta["intensity"]["min"] + meta["windowWidth"] / 2

        if window_width is None:
            if "windowWidth" in meta:
                window_width = meta["windowWidth"]
            else:
                window_width = meta["intensity"]["max"] - meta["intensity"]["min"]

        data = {
            "volumeId": volume_id,
            "sliceIndex": slice_index,
            "normal": normal,
            "windowCenter": window_center,
            "windowWidth": window_width,
        }
        image_bytes = self._api.post(
            method="volumes.slices.images.download", data=data, stream=True
        ).content

        return read_bytes(image_bytes)
