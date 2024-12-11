import cv2
import numpy as np


def one_hot(segmentation, num_classes):
    # TODO: this assumes that the classes start with 0
    segmentation_copy = segmentation.copy()
    eye = np.eye(num_classes + 1, dtype=np.bool_)
    one_hot_seg = eye[segmentation_copy].transpose(2, 0, 1)
    return one_hot_seg[:num_classes]


def single_one_hot(segmentation, cls):
    one_hot_seg = np.zeros_like(segmentation)
    one_hot_seg[segmentation == cls] = 1
    return one_hot_seg


def get_single_contiguous_segment(one_hot_segmentation):
    import scipy  # pylint: disable=import-error

    kernel = np.ones((3, 3), dtype=one_hot_segmentation.dtype)
    seg = scipy.ndimage.label(one_hot_segmentation, structure=kernel)[0]
    return [np.where(seg == l) for l in range(1, seg.max() + 1)]


def get_contiguous_segments(one_hot_segmentation):
    import scipy  # pylint: disable=import-error

    kernel = np.ones((3, 3), dtype=one_hot_segmentation.dtype)
    segments_tensor = np.stack(
        [scipy.ndimage.label(seg, structure=kernel)[0] for seg in one_hot_segmentation]
    )

    return {
        c: [np.where(seg == l) for l in range(1, seg.max() + 1)]
        for c, seg in enumerate(segments_tensor)
    }


def get_exact_kernel(width):
    x = np.arange(-width, width + 1)
    xx, yy = np.meshgrid(x, x)
    d = np.sqrt(xx**2 + yy**2)
    kernel = (np.round(d) <= width).astype(np.uint8)
    return kernel


def erode_mask(mask, width, implementation="fast"):
    if mask.ndim == 2:
        input_mask = mask.astype(np.uint8)
    elif mask.ndim == 3:
        input_mask = mask.astype(np.uint8).transpose(1, 2, 0)
    else:
        raise ValueError(f"Assume an array of shape (H,W) or (C,H,W), got {mask.shape}!")

    if implementation == "fast":
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded_mask = cv2.erode(input_mask, kernel, iterations=width)
    elif implementation == "exact":
        kernel = get_exact_kernel(width)
        eroded_mask = cv2.erode(input_mask, kernel, iterations=1)
    else:
        raise ValueError(
            f'Implementation has to be one of "exact" and "fast", received: {implementation}!'
        )

    if input_mask.ndim == 2:
        return eroded_mask
    else:  # input_mask.ndim == 3
        if mask.shape[0] == 1:  # in this case, the first axis gets removed by cv2
            eroded_mask = np.expand_dims(eroded_mask, -1)
        return eroded_mask.transpose(2, 0, 1)


def dilate_mask(mask, width, implementation="fast"):
    if mask.ndim == 2:
        input_mask = mask.astype(np.uint8)
    elif mask.ndim == 3:
        input_mask = mask.astype(np.uint8).transpose(1, 2, 0)
    else:
        raise ValueError(f"Assume an array of shape (H,W) or (C,H,W), got {mask.shape}!")

    if implementation == "fast":
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated_mask = cv2.dilate(input_mask, kernel, iterations=width)
    elif implementation == "exact":
        kernel = get_exact_kernel(width)
        dilated_mask = cv2.dilate(input_mask, kernel, iterations=1)
    else:
        raise ValueError(
            f'Implementation has to be one of "exact" and "fast", received: {implementation}!'
        )

    if input_mask.ndim == 2:
        return dilated_mask
    else:  # input_mask.ndim == 3
        if mask.shape[0] == 1:  # in this case, the first axis gets removed by cv2
            dilated_mask = np.expand_dims(dilated_mask, -1)
        return dilated_mask.transpose(2, 0, 1)


def get_interior_boundary(mask, width, implementation="fast"):
    eroded_mask = erode_mask(mask, width, implementation)
    boundary_mask = (mask - eroded_mask).astype(np.bool_)
    return boundary_mask


def get_exterior_boundary(mask, width, implementation="fast"):
    dilated_mask = dilate_mask(mask, width, implementation)
    boundary_mask = (dilated_mask - mask).astype(np.bool_)
    return boundary_mask
