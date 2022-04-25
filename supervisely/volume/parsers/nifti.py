import math

from supervisely.volume.loaders import nifti


def load(entry_path):
    vol = nifti.open_image(entry_path, verbose=False)
    vol_info = {}

    header = vol.src_object.header

    # can dim[0] >= 5 and not multi channels with RGB datatype?

    if header["dim"][0] >= 5:
        vol_info["channelsCount"] = header["dim"][5]
    elif header["datatype"] == 128:
        vol_info["channelsCount"] = 3
    elif header["datatype"] == 2304:
        vol_info["channelsCount"] = 4

    if not math.isnan(header["scl_slope"]):
        vol_info["rescaleSlope"] = float(header["scl_slope"])

    if not math.isnan(header["scl_inter"]):
        vol_info["rescaleIntercept"] = float(header["scl_inter"])

    return [(None, vol, vol_info)]
