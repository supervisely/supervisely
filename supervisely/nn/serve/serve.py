from supervisely._utils import is_production, is_development


class Serve:
    def __init__(self, dir: str):
        if is_production():
            raise NotImplementedError("TBD - download directory")
        elif is_development():
            pass
