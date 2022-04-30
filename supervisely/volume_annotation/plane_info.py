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
