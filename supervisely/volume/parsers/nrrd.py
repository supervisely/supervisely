from supervisely.volume.loaders import nrrd


def load(entry_path):
    vol = nrrd.open_image(entry_path, verbose=False)
    vol_info = {}

    return [(None, vol, vol_info)]
