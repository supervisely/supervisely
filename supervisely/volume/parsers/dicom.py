from pydicom.valuerep import MultiValue
import pydicom
import os
import supervisely as sly
from supervisely import TaskPaths
from supervisely.volume.loaders.dicom import SliceStacker
import magic

photometricInterpretationRGB = set(
    [
        "RGB",
        "PALETTE COLOR",
        "YBR_FULL",
        "YBR_FULL_422",
        "YBR_PARTIAL_422",
        "YBR_PARTIAL_420",
        "YBR_RCT",
    ]
)


def unpack_value(val):
    if isinstance(val, MultiValue):
        return val[0]
    return val


def load(entry_path):
    suids = set([])
    for file in os.listdir(entry_path):
        full_path = os.path.join(entry_path, file)
        if (
            file.lower().endswith("dcm")
            or magic.from_file(full_path, mime=True) == "application/dicom"
        ):
            si_uid = pydicom.read_file(full_path, stop_before_pixels=True)[
                SliceStacker.SI_UID_TAG
            ].value
            suids.add(si_uid)

    results = []
    for si_uid in suids:
        try:
            vol = SliceStacker(entry_path, si_uid=si_uid).execute().volume
            vol_info = {}

            slices = list(vol.src_object.slices.values())
            slice = slices[0]

            vol_info["seriesInstanceUID"] = slice.SeriesInstanceUID

            if hasattr(slice, "Modality"):
                vol_info["modality"] = slice.Modality

            if hasattr(slice, "PatientID"):
                vol_info["patientId"] = slice.PatientID

            if hasattr(slice, "PatientName"):
                vol_info["patientName"] = str(slice.PatientName)

            if hasattr(slice, "WindowCenter"):
                vol_info["windowCenter"] = float(unpack_value(slice.WindowCenter))

            if hasattr(slice, "WindowWidth"):
                vol_info["windowWidth"] = float(unpack_value(slice.WindowWidth))

            if hasattr(slice, "RescaleIntercept"):
                vol_info["rescaleIntercept"] = float(slice.RescaleIntercept)

            if hasattr(slice, "RescaleSlope"):
                vol_info["rescaleSlope"] = float(slice.RescaleSlope)

            if (
                hasattr(slice, "PhotometricInterpretation")
                and slice.PhotometricInterpretation in photometricInterpretationRGB
            ):
                vol_info["channelsCount"] = 3

            dataset_name = si_uid
            folder_name = os.path.basename(entry_path.replace(TaskPaths.DATA_DIR, ""))
            if folder_name:
                dataset_name = "{} ({})".format(folder_name, si_uid)

            results.append((dataset_name, vol, vol_info))
        except Exception as e:
            sly.logger.warn(
                "Series {} skipped due to error: {}".format(si_uid, str(e)),
                exc_info=True,
                extra={
                    "exc_str": str(e),
                    "file_path": entry_path,
                },
            )

    return results
