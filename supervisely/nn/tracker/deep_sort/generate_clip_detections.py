import numpy as np
import torch  # pylint: disable=import-error
from PIL import Image


def extract_image_patch(image, bbox, patch_shape=None):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    # image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, model, transform, device):

        self.model = model
        self.transform = transform
        self.device = str(device)

    def __call__(self, data_x, batch_size=32):
        out = []

        for patch in range(len(data_x)):
            if self.device == "cpu":
                img = self.transform(Image.fromarray(data_x[patch]))
            else:
                img = self.transform(Image.fromarray(data_x[patch])).cuda()
            out.append(img)

        features = self.model.encode_image(torch.stack(out)).cpu().detach().numpy()
        return features


def create_box_encoder(model, transform, batch_size=32, device="cpu"):
    image_encoder = ImageEncoder(model, transform, device)

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            # print("extracting box {} from image {}".format(box, image.shape))
            patch = extract_image_patch(image, box)
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(0.0, 255.0, image.shape).astype(np.uint8)
            image_patches.append(patch)
        # image_patches = np.array(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder
