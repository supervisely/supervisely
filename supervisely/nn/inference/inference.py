from supervisely._utils import is_production, is_development


class Inference:
    def __init__(self, model_dir: str = None):
        if is_production():
            raise NotImplementedError("TBD - download directory")
        elif is_development():
            pass

    @property
    def model_meta(self):
        raise NotImplementedError()

    def apply_to_image(
        id,
        inplace=False,
        replce_labels=False,
    ):
        raise NotImplementedError()

    def apply_to_project(id):
        raise NotImplementedError()

    def apply_to_dataset(id):
        raise NotImplementedError()

    def apply_to_video(id):
        raise NotImplementedError()
