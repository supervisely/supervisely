# coding: utf-8
import numpy as np

# pylint: disable=import-error
import torch
import torch.nn.functional as torch_functional
# pylint: enable=import-error

from supervisely.imaging import image as sly_image
from supervisely.nn.legacy.pytorch.cuda import cuda_variable


def infer_per_pixel_scores_single_image(model, raw_input, out_shape, apply_softmax=True):
    """
    Performs inference with PyTorch model and resize predictions to a given size.

    :param model: PyTorch model inherited from torch.Module class.
    :type model: :class:`~torch.nn.Module`
    :param raw_input: PyTorch Tensor
    :type raw_input: :class:`~torch.Tensor`
    :param out_shape: Output size (height, width).
    :type out_shape: Tuple[int, int]
    :param apply_softmax: Whether to apply softmax function after inference or not.
    :type apply_softmax: bool
    :returns: Inference resulting numpy array resized to a given size.
    :rtype: np.ndarray
    """
    model_input = torch.stack([raw_input], 0)  # add dim #0 (batch size 1)
    model_input = cuda_variable(model_input, volatile=True)

    output = model(model_input)
    if apply_softmax:
        output = torch_functional.softmax(output, dim=1)
    output = output.data.cpu().numpy()[0]  # from batch to 3d

    pred = np.transpose(output, (1, 2, 0))
    return sly_image.resize(pred, out_shape)
