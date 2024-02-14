import json
import os
import xml.etree.ElementTree as ET

from pycocotools.coco import COCO

from supervisely.annotation.annotation import Annotation
from supervisely.convert.base_format import BaseFormat
from supervisely.convert.image.coco.coco_converter import COCOFormat
from supervisely.convert.image.pascal_voc.pascal_voc_converter import PascalVOCFormat
from supervisely.convert.image.sly.sly_image_converter import SLYImageFormat
from supervisely.convert.image.yolo.yolo_converter import YOLOFormat
from supervisely.io.fs import get_file_ext
from supervisely.io.json import load_json_file

ALLOWED_IMAGE_ANN_EXTENSIONS = [".json", ".txt", ".xml"]

class ImageFormatConverter:
    def __init__(self, input_data):
        self.input_data = input_data
        self.converter = self._detect_format()

    @property
    def format(self):
        return self.converter.format
    
    
    def _detect_format(self):
        # Filter out formats with counts above 0
        format_counts = self._get_formats()
        valid_formats = {fmt: count for fmt, count in format_counts.items() if count > 0}

        # Check if more than one format is present
        if len(valid_formats) > 1:
            raise ValueError("Mixed annotation formats are not supported.")
        elif len(valid_formats) == 0:
            # raise ValueError("No valid annotation formats were found.")
            return None
        else:
            # Only one valid format detected
            format_name = list(valid_formats.keys())[0]
            if format_name == "Supervisely":
                return SLYImageFormat(self.input_data)
            elif format_name == "COCO":
                return COCOFormat(self.input_data)
            elif format_name == "Pascal VOC":
                return PascalVOCFormat(self.input_data)
            elif format_name == "YOLO":
                return YOLOFormat(self.input_data)
            else:
                raise ValueError(f"Unsupported annotation format: {format_name}")
            
    def _get_formats(self):
        """return converter class"""
        format_counts = {'YOLO': 0, 'COCO': 0, 'Pascal VOC': 0, 'Supervisely': 0, 'Unknown': 0}
        for root, _, files in os.walk(self.input_data):
            for file in files:
                ann_path = os.path.join(root, file)
                ext = get_file_ext(ann_path)
                if ext in ALLOWED_IMAGE_ANN_EXTENSIONS:
                    format_detected = self._validate_file_format(ann_path)
                    if format_detected:
                        format_counts[format_detected] += 1
                    else:
                        format_counts['Unknown'] += 1
        return format_counts
    
    def _validate_file_format(self, file_path):
        try:
            # YOLO check
            if file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5 or not all(part.replace('.', '', 1).isdigit() for part in parts):
                            return False
                return 'YOLO'
            elif file_path.endswith('.json'):
                try:
                    # COCO check
                    ann = COCO(file_path)
                    return 'COCO'
                except:
                    pass
                
                try:
                    # Supervisely check
                    data = load_json_file(file_path)
                    ann = Annotation.from_json(data)
                    return 'Supervisely'
                except:
                    return None
            
            # Pascal VOC check
            elif file_path.endswith('.xml'):
                tree = ET.parse(file_path)
                root = tree.getroot()
                if root.tag == 'annotation':
                    return 'Pascal VOC'
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        return None

