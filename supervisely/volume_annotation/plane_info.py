class PlaneName:
    SAGITTAL = "sagittal"
    CORONAL = "coronal"
    AXIAL = "axial"
    _valid_names = [SAGITTAL, CORONAL, AXIAL]

    @staticmethod
    def validate(name):
        if name not in PlaneName._valid_names:
            raise ValueError(
                f"Unknown plane {name}, valid names are {PlaneName._valid_names}"
            )


def get_normal(name):
    PlaneName.validate(name)
    if name == PlaneName.SAGITTAL:
        return {"x": 1, "y": 0, "z": 0}
    if name == PlaneName.CORONAL:
        return {"x": 0, "y": 1, "z": 0}
    if name == PlaneName.AXIAL:
        return {"x": 0, "y": 0, "z": 1}


# "volumeMeta": {
#     "ACS": "RAS",
#     "intensity": { "max": 3071, "min": -3024 },
#     "windowWidth": 6095,
#     "rescaleSlope": 1,
#     "windowCenter": 23.5,
#     "channelsCount": 1,
#     "dimensionsIJK": { "x": 512, "y": 512, "z": 139 },
#     "IJK2WorldMatrix": [
#         0.7617189884185793, 0, 0, -194.238403081894, 0, 0.7617189884185793, 0,
#         -217.5384061336518, 0, 0, 2.5, -347.7500000000001, 0, 0, 0, 1
#     ],
#     "rescaleIntercept": 0
# },


def get_img_size_from_volume_meta(name, volume_meta):
    PlaneName.validate(name)
    dimentions = volume_meta["dimensionsIJK"]
    # (height, width)
    height = None
    width = None
    if name == PlaneName.SAGITTAL:
        width = dimentions["y"]
        height = dimentions["z"]
    elif name == PlaneName.CORONAL:
        width = dimentions["x"]
        height = dimentions["z"]
    elif name == PlaneName.AXIAL:
        width = dimentions["x"]
        height = dimentions["y"]
    return [height, width]


def get_slices_count_from_volume_meta(name, volume_meta):
    PlaneName.validate(name)
    dimentions = volume_meta["dimensionsIJK"]
    if name == PlaneName.SAGITTAL:
        return dimentions["x"]
    elif name == PlaneName.CORONAL:
        return dimentions["y"]
    elif name == PlaneName.AXIAL:
        return dimentions["y"]
