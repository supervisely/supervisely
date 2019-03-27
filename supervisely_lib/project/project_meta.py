# coding: utf-8

from typing import List

from supervisely_lib.io.json import JsonSerializable
from supervisely_lib.annotation.obj_class_collection import ObjClassCollection
from supervisely_lib.annotation.tag_meta_collection import TagMetaCollection
from supervisely_lib.annotation.obj_class import ObjClass
from supervisely_lib.annotation.tag_meta import TagMeta


class ProjectMetaJsonFields:
    OBJ_CLASSES = 'classes'
    IMG_TAGS = 'tags_images'
    OBJ_TAGS = 'tags_objects'


#@TODO: add validation
class ProjectMeta(JsonSerializable):
    def __init__(self, obj_classes=None, img_tag_metas=None, obj_tag_metas=None):
        self._obj_classes = ObjClassCollection() if obj_classes is None else obj_classes
        # TODO do we actualy need two sets of tags?
        self._img_tag_metas = TagMetaCollection() if img_tag_metas is None else img_tag_metas
        self._obj_tag_metas = TagMetaCollection() if obj_tag_metas is None else obj_tag_metas

    @property
    def obj_classes(self):
        return self._obj_classes

    @property
    def img_tag_metas(self):
        return self._img_tag_metas

    @property
    def obj_tag_metas(self):
        return self._obj_tag_metas

    def to_json(self):
        return {
            ProjectMetaJsonFields.OBJ_CLASSES: self._obj_classes.to_json(),
            ProjectMetaJsonFields.IMG_TAGS: self._img_tag_metas.to_json(),
            ProjectMetaJsonFields.OBJ_TAGS: self._obj_tag_metas.to_json()
        }

    @classmethod
    def from_json(cls, data):
        return cls(ObjClassCollection.from_json(data[ProjectMetaJsonFields.OBJ_CLASSES]),
                   TagMetaCollection.from_json(data[ProjectMetaJsonFields.IMG_TAGS]),
                   TagMetaCollection.from_json(data[ProjectMetaJsonFields.OBJ_TAGS]))

    def merge(self, other):
        return self.clone(obj_classes=self._obj_classes.merge(other.obj_classes),
                          img_tag_metas=self._img_tag_metas.merge(other.img_tag_metas),
                          obj_tag_metas=self._obj_tag_metas.merge(other.obj_tag_metas))

    def clone(self, obj_classes: ObjClassCollection = None, img_tag_metas: TagMetaCollection = None, obj_tag_metas: TagMetaCollection = None):
        return ProjectMeta(obj_classes=obj_classes or self.obj_classes,
                           img_tag_metas=img_tag_metas or self.img_tag_metas,
                           obj_tag_metas=obj_tag_metas or self.obj_tag_metas)

    def add_obj_class(self, new_obj_class):
        return self.add_obj_classes([new_obj_class])

    def add_obj_classes(self, new_obj_classes):
        return self.clone(obj_classes=self.obj_classes.add_items(new_obj_classes))

    def add_img_tag_meta(self, new_tag_meta):
        return self.add_img_tag_metas([new_tag_meta])

    def add_img_tag_metas(self, new_tag_metas):
        return self.clone(img_tag_metas=self.img_tag_metas.add_items(new_tag_metas))

    def add_obj_tag_meta(self, new_tag_meta):
        return self.add_obj_tag_metas([new_tag_meta])

    def add_obj_tag_metas(self, new_tag_metas):
        return self.clone(obj_tag_metas=self.obj_tag_metas.add_items(new_tag_metas))

    @staticmethod
    def _delete_items(collection, item_names):
        names_to_delete = set(item_names)
        res_items = []
        for item in collection:
            if item.key() not in names_to_delete:
                res_items.append(item)
        return res_items

    def delete_obj_class(self, obj_class_name):
        return self.delete_obj_classes([obj_class_name])

    def delete_obj_classes(self, obj_class_names):
        res_items = self._delete_items(self._obj_classes, obj_class_names)
        return self.clone(obj_classes=ObjClassCollection(res_items))

    def delete_img_tag_meta(self, tag_name):
        self.delete_img_tag_metas([tag_name])

    def delete_img_tag_metas(self, tag_names):
        res_items = self._delete_items(self._img_tag_metas, tag_names)
        return self.clone(img_tag_metas=TagMetaCollection(res_items))

    def delete_obj_tag_meta(self, tag_name):
        self.delete_obj_tag_metas([tag_name])

    def delete_obj_tag_metas(self, tag_names):
        res_items = self._delete_items(self._obj_tag_metas, tag_names)
        return self.clone(obj_tag_metas=TagMetaCollection(res_items))

    def get_obj_class(self, obj_class_name):
        return self._obj_classes.get(obj_class_name)

    def get_img_tag_meta(self, tag_name):
        return self._img_tag_metas.get(tag_name)

    def get_obj_tag_meta(self, tag_name):
        return self._obj_tag_metas.get(tag_name)

    @staticmethod
    def merge_list(metas):
        res_meta = ProjectMeta()
        for meta in metas:
            res_meta = res_meta.merge(meta)
        return res_meta

    def __str__(self):
        result = 'ProjectMeta:\n'
        result += 'Object Classes\n{}\n'.format(str(self._obj_classes))
        result += 'Image Tags\n{}\n'.format(str(self._img_tag_metas))
        result += 'Object Tags\n{}\n'.format(str(self._obj_tag_metas))
        return result