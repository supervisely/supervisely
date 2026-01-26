from typing import Dict, Any
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
        # add to dataset
        self.samples[image_id] = sample
        self.samples_list.append(sample)
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

    def _save_mask(self, annotation: sly.Annotation, image_name: str) -> str:

        mapping = {label.obj_class: label.obj_class for label in annotation.labels}
        ann_nonoverlap = annotation.to_nonoverlapping_masks(mapping)
        h, w = annotation.img_size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for label in ann_nonoverlap.labels:
            class_name = label.obj_class.name
            class_id = self.class2idx.get(class_name)
            if class_id is not None:
                label.geometry.draw(mask, color=class_id)
        
        mask_name = Path(image_name).stem + '.png' 
        mask_path = str(self.masks_dir / mask_name)
        cv2.imwrite(mask_path, mask)
        
        return mask_path

    def __len__(self) -> int:
        return len(self.samples)
    
    def get_image_ids(self) -> list:
        """Get list of image IDs in dataset"""
        return list(self.samples.keys())
    