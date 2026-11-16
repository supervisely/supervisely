from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from PIL import Image
import numpy as np
import supervisely as sly
import cv2


class IncrementalDataset:
    """
    1. Save images on disk
    2. Store annotations in SLY/COCO format. Handle case for Segmentation task
    3. Implement indexing, adding, and updating samples
    4. __getitem__ to retrieve samples by index
    """
    def __init__(
            self,
            class2idx: dict,
            data_dir: str,
            save_masks_as_images: bool = False,
        ):
        """
        :param class2idx: Class name to index.
        :type class2idx: dict
        :param data_dir: Root data directory.
        :type data_dir: str
        :param save_masks_as_images: Save masks as images.
        :type save_masks_as_images: bool
        """
        self.class2idx = class2idx
        self.data_dir = Path(data_dir)
        self.save_masks_as_images = save_masks_as_images
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if self.save_masks_as_images:
            self.masks_dir = self.data_dir / "masks"
            self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.samples: Dict[int, dict] = {}
        self.samples_list = []
        # image_id for each index of samples_list
        self._ids_list: List[int] = []
        # image_ids that were yielded by the sampler at least once (i.e. trained on)
        self.sampled_image_ids: Set[int] = set()
        # Mask rasterization settings (semantic segmentation)
        self.ignore_class_names: Set[str] = set()
        self.class_priority: Optional[List[str]] = None
        self.ignore_index = 255

    def update_class_map(self, class2idx: dict):
        self.class2idx = class2idx

    def set_rasterization(
            self,
            ignore_class_names: Optional[List[str]] = None,
            class_priority: Optional[List[str]] = None,
        ):
        """
        Configure how masks are rasterized.

        :param ignore_class_names: Classes painted with ignore_index (255) instead of being learned.
        :type ignore_class_names: List[str], optional
        :param class_priority: Class names from the highest priority (drawn on top) to the lowest.
            If None, the object order in the annotation is used: a later object wins.
        :type class_priority: List[str], optional
        """
        self.ignore_class_names = set(ignore_class_names or [])
        self.class_priority = list(class_priority) if class_priority else None

    def mark_sampled(self, idx: int):
        """Called by the sampler when the sample at idx is taken for training."""
        if 0 <= idx < len(self._ids_list):
            self.sampled_image_ids.add(self._ids_list[idx])

    def add(
            self,
            image_id: int,
            image_np: np.ndarray,
            annotation: sly.Annotation,
            image_name: str
        ) -> dict:
        if image_id in self.samples:
            raise ValueError(f"Cannot add sample: Image ID {image_id} already exists in the dataset.")
        image_name = f"{image_id} {image_name}"
        w, h = image_np.shape[1], image_np.shape[0]
        img_size = (w, h)
        image_path = self._save_img(image_np, image_name)
        mask_path = None
        if self.save_masks_as_images:
            mask_path = self._save_mask(annotation, image_name)
        sample = self._format_sample(
            image_id,
            annotation,
            img_size,
            image_path,
            mask_path
        )
        assert isinstance(sample, dict), "Sample must be a dict."
        # add extra fields for internal use
        sample['image_path'] = image_path
        sample['size'] = img_size
        if mask_path is not None:
            sample['mask_path'] = mask_path
        # add to dataset. __len__ reads self.samples, so it is updated last:
        # the sampler (training thread) must never see an index missing from samples_list.
        self._ids_list.append(image_id)
        self.samples_list.append(sample)
        self.samples[image_id] = sample
        return sample

    def update(
            self,
            image_id: int,
            annotation: sly.Annotation,
        ) -> dict:
        if image_id not in self.samples:
            raise ValueError(f"Cannot update sample: Image ID {image_id} does not exist in the dataset.")
        sample = self.samples[image_id]
        new_sample = self._format_sample(
            image_id,
            annotation,
            sample['size'],
            sample['image_path'],
            sample.get('mask_path')
        )
        sample.update(new_sample)
        return sample

    def add_or_update(
            self,
            image_id: int,
            image_np: np.ndarray,
            annotation: sly.Annotation,
            image_name: str
        ) -> dict:
        if image_id not in self.samples:
            return self.add(image_id, image_np, annotation, image_name)
        else:
            return self.update(image_id, annotation)

    def _format_sample(
            self,
            image_id: int,
            annotation: sly.Annotation, 
            image_size: tuple,
            image_path: str,
            mask_path: str = None
        ) -> dict:
        # Ignored (unselected) classes only matter for mask rasterization
        trained_labels = [label for label in annotation.labels if label.obj_class.name in self.class2idx]
        if len(trained_labels) != len(annotation.labels):
            annotation = annotation.clone(labels=trained_labels)
        sample = {
            'image_id': image_id,
            'width': image_size[0],
            'height': image_size[1],
            'annotations': annotation.to_coco(annotation.image_id, self.class2idx)[0],
            'image_path': image_path,
            'mask_path': mask_path
        }
        return sample

    def _save_img(self, image_np: np.ndarray, image_name: str) -> str:
        image = Image.fromarray(image_np).convert('RGB')
        image_path = str(self.images_dir / image_name)
        image.save(image_path)
        return image_path

    def _sort_labels_for_rasterization(self, labels: list) -> list:
        """
        Order labels so that the label drawn last wins where masks overlap.

        Without class priority the annotation order is kept (a later object wins).
        With class priority ignored classes go to the bottom, then selected classes
        from the lowest to the highest priority. The sort is stable, so the object
        order is kept within a class.
        """
        if not self.class_priority:
            return list(labels)
        rank = {name: i for i, name in enumerate(reversed(self.class_priority))}

        def key(label):
            name = label.obj_class.name
            if name in self.ignore_class_names:
                return -1
            return rank.get(name, -1)

        return sorted(labels, key=key)

    def _rasterize_mask(self, annotation: sly.Annotation) -> np.ndarray:
        labels = self._sort_labels_for_rasterization(annotation.labels)
        annotation = annotation.clone(labels=labels)
        # to_nonoverlapping_masks turns every label into a Bitmap, so the destination class must be
        # a Bitmap class too (otherwise Polygon classes fail label geometry validation).
        mapping = {
            label.obj_class: label.obj_class.clone(geometry_type=sly.Bitmap)
            for label in annotation.labels
        }
        ann_nonoverlap = annotation.to_nonoverlapping_masks(mapping)
        h, w = annotation.img_size
        mask = np.zeros((h, w), dtype=np.uint8)

        for label in ann_nonoverlap.labels:
            class_name = label.obj_class.name
            if class_name in self.ignore_class_names:
                class_id = self.ignore_index
            else:
                class_id = self.class2idx.get(class_name)
            if class_id is not None:
                label.geometry.draw(mask, color=class_id)
        return mask

    def _save_mask(self, annotation: sly.Annotation, image_name: str) -> str:
        mask = self._rasterize_mask(annotation)
        mask_name = Path(image_name).stem + '.png'
        mask_path = str(self.masks_dir / mask_name)
        cv2.imwrite(mask_path, mask)

        return mask_path

    def __len__(self) -> int:
        return len(self.samples)

    def get_image_ids(self) -> list:
        """Get list of image IDs in dataset"""
        return list(self.samples.keys())
