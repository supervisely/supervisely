from typing import List, Union
import supervisely as sly


class ClassMap:
    def __init__(self, obj_classes: Union[sly.ObjClassCollection, List[sly.ObjClass]]):
        self.obj_classes = obj_classes
        self.class2idx = {obj_class.name: idx for idx, obj_class in enumerate(self.obj_classes)}
        self.idx2class = {idx: obj_class.name for idx, obj_class in enumerate(self.obj_classes)}
        self.classes = [obj_class.name for obj_class in self.obj_classes]
        self.sly_ids = [obj_class.sly_id for obj_class in self.obj_classes]
    
    def __len__(self):
        return len(self.obj_classes)