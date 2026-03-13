# coding: utf-8


class SingleImageInferenceInterface:
    """Abstract interface for single-image inference; subclasses implement inference() and get_out_meta()."""

    def inference(self, image, ann):
        raise NotImplementedError()

    def get_out_meta(self):
        raise NotImplementedError()
