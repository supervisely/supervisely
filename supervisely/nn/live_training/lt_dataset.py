from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import supervisely as sly
import json

class LTDataset:
    def __init__(self, images_dir: str, classes: List[str]):
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.classes = classes
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        
        self.samples = {}
        self._image_ids = []
    
    def __len__(self) -> int:
        return len(self._image_ids)
    
    def __contains__(self, image_id: int) -> bool:
        return image_id in self.samples
    
    def add_or_update(
        self,
        image_np: np.ndarray,
        annotation: dict,
        image_info: dict,
        project_meta: sly.ProjectMeta
    ) -> int:
        image_id = image_info['image_id']
        
        # Save image
        filename = f"img_{image_id}.jpg"
        image = Image.fromarray(image_np).convert('RGB')
        image_path = self.images_dir / filename
        image.save(image_path)
        
        height, width = image_np.shape[:2]
        
        # Convert SLY to COCO
        sly_ann = sly.Annotation.from_json(annotation, project_meta)
        coco_annotations, _ = sly_ann.to_coco(
            coco_image_id=image_id,
            class_mapping=self.class_to_idx
        )
        
        # Store
        self.samples[image_id] = {
            'file_name': filename,
            'width': width,
            'height': height,
            'annotations': coco_annotations,
        }
        
        if image_id not in self._image_ids:
            self._image_ids.append(image_id)
        
        return image_id
    
    def get_image_ids(self) -> List[int]:
        return self._image_ids.copy()
    
    def save_coco_json(self, output_path: str) -> None:
        images = []
        annotations = []
        
        for image_id in self._image_ids:
            sample = self.samples[image_id]
            
            images.append({
                'id': image_id,
                'file_name': sample['file_name'],
                'width': sample['width'],
                'height': sample['height'],
            })
            
            annotations.extend(sample['annotations'])
        
        categories = [
            {'id': idx, 'name': name, 'supercategory': name}
            for name, idx in self.class_to_idx.items()
        ]
        
        coco_dict = {
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }
        
        
        with open(output_path, 'w') as f:
            json.dump(coco_dict, f, indent=2)



