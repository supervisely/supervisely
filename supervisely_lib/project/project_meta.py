# coding: utf-8

from supervisely_lib.io.json import JsonSerializable
from supervisely_lib.annotation.obj_class_collection import ObjClassCollection
from supervisely_lib.annotation.tag_meta_collection import TagMetaCollection
from supervisely_lib._utils import take_with_default


class ProjectMetaJsonFields:
    OBJ_CLASSES = 'classes'
    IMG_TAGS = 'tags_images'
    OBJ_TAGS = 'tags_objects'
    TAGS = 'tags'


def _merge_img_obj_tag_metas(img_tag_metas: ObjClassCollection,
                             obj_tag_metas: ObjClassCollection) -> ObjClassCollection:
    obj_tag_metas_to_add = []
    for obj_tag_meta in obj_tag_metas:
        img_tag_meta_same_key = img_tag_metas.get(obj_tag_meta.key(), None)
        if img_tag_meta_same_key is None:
            obj_tag_metas_to_add.append(obj_tag_meta)
        elif not img_tag_meta_same_key.is_compatible(obj_tag_meta):
            raise ValueError(
                'Unable to merge tag metas for images and objects. Found tags with the same name, but incompatible '
                'values. \n Image-level tag meta: {}\n Object-level tag meta: {}.\n Rename one of the tags to have a '
                'unique name to be able to load project meta.'.format(str(img_tag_meta_same_key), str(obj_tag_meta)))
    return img_tag_metas.add_items(obj_tag_metas_to_add)


#@TODO: add validation
class ProjectMeta(JsonSerializable):
    def __init__(self, obj_classes=None, tag_metas=None):
        self._obj_classes = ObjClassCollection() if obj_classes is None else obj_classes
        self._tag_metas = take_with_default(tag_metas, TagMetaCollection())

    @property
    def obj_classes(self):
        return self._obj_classes

    @property
    def tag_metas(self):
        return self._tag_metas

    def to_json(self):
        return {
            ProjectMetaJsonFields.OBJ_CLASSES: self._obj_classes.to_json(),
            ProjectMetaJsonFields.TAGS: self._tag_metas.to_json(),
        }

    @classmethod
    def from_json(cls, data):
        tag_metas_json = data.get(ProjectMetaJsonFields.TAGS, [])
        img_tag_metas_json = data.get(ProjectMetaJsonFields.IMG_TAGS, [])
        obj_tag_metas_json = data.get(ProjectMetaJsonFields.OBJ_TAGS, [])

        if len(tag_metas_json) > 0:
            # New format - all project tags in a single collection.
            if any(len(x) > 0 for x in [img_tag_metas_json, obj_tag_metas_json]):
                raise ValueError(
                    'Project meta JSON contains both the {!r} section (current format merged tags for all of '
                    'the project) and {!r} or {!r} sections (legacy format with separate collections for images '
                    'and labeled objects). Either new format only or legacy format only are supported, but not a '
                    'mix.'.format(
                        ProjectMetaJsonFields.TAGS, ProjectMetaJsonFields.IMG_TAGS, ProjectMetaJsonFields.OBJ_TAGS))
            tag_metas = TagMetaCollection.from_json(tag_metas_json)
        else:
            img_tag_metas = TagMetaCollection.from_json(img_tag_metas_json)
            obj_tag_metas = TagMetaCollection.from_json(obj_tag_metas_json)
            tag_metas = _merge_img_obj_tag_metas(img_tag_metas, obj_tag_metas)

        return cls(obj_classes=ObjClassCollection.from_json(data[ProjectMetaJsonFields.OBJ_CLASSES]),
                   tag_metas=tag_metas)

    def merge(self, other):
        return self.clone(obj_classes=self._obj_classes.merge(other.obj_classes),
                          tag_metas=self._tag_metas.merge(other._tag_metas))

    def clone(self, obj_classes: ObjClassCollection = None, tag_metas: TagMetaCollection = None):
        return ProjectMeta(obj_classes=take_with_default(obj_classes, self.obj_classes),
                           tag_metas=take_with_default(tag_metas, self.tag_metas))

    def add_obj_class(self, new_obj_class):
        return self.add_obj_classes([new_obj_class])

    def add_obj_classes(self, new_obj_classes):
        return self.clone(obj_classes=self.obj_classes.add_items(new_obj_classes))

    def add_tag_meta(self, new_tag_meta):
        return self.add_tag_metas([new_tag_meta])

    def add_tag_metas(self, new_tag_metas):
        return self.clone(tag_metas=self.tag_metas.add_items(new_tag_metas))

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

    def delete_tag_meta(self, tag_name):
        return self.delete_tag_metas([tag_name])

    def delete_tag_metas(self, tag_names):
        res_items = self._delete_items(self._tag_metas, tag_names)
        return self.clone(tag_metas=TagMetaCollection(res_items))

    def get_obj_class(self, obj_class_name):
        return self._obj_classes.get(obj_class_name)

    def get_tag_meta(self, tag_name):
        return self._tag_metas.get(tag_name)

    @staticmethod
    def merge_list(metas):
        res_meta = ProjectMeta()
        for meta in metas:
            res_meta = res_meta.merge(meta)
        return res_meta

    def __str__(self):
        result = 'ProjectMeta:\n'
        result += 'Object Classes\n{}\n'.format(str(self._obj_classes))
        result += 'Tags\n{}\n'.format(str(self._tag_metas))
        return result