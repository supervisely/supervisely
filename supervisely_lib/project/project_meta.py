# coding: utf-8
from __future__ import annotations

from supervisely_lib.io.json import JsonSerializable
from supervisely_lib.annotation.obj_class_collection import ObjClassCollection
from supervisely_lib.annotation.tag_meta_collection import TagMetaCollection
from supervisely_lib._utils import take_with_default

from supervisely_lib.annotation.obj_class import ObjClass
from supervisely_lib.geometry.polygon import Polygon
from supervisely_lib.geometry.bitmap import Bitmap
from supervisely_lib.geometry.rectangle import Rectangle


class ProjectMetaJsonFields:
    OBJ_CLASSES = 'classes'
    IMG_TAGS = 'tags_images'
    OBJ_TAGS = 'tags_objects'
    TAGS = 'tags'
    PROJECT_TYPE = 'projectType'


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


class ProjectMeta(JsonSerializable):
    '''
    This is a class for creating and using ProjectMeta objects. This class contain data about meta information of the project
    '''
    def __init__(self, obj_classes=None, tag_metas=None, project_type=None):
        '''
        :param obj_classes: Collection that stores ObjClass instances with unique names.
        :param tag_metas: Collection that stores TagMeta instances with unique names.
        '''
        self._obj_classes = ObjClassCollection() if obj_classes is None else obj_classes
        self._tag_metas = take_with_default(tag_metas, TagMetaCollection())
        self._project_type = project_type

    @property
    def obj_classes(self) -> ObjClassCollection:
        return self._obj_classes

    @property
    def tag_metas(self) -> TagMetaCollection:
        return self._tag_metas

    @property
    def project_type(self):
        return self._project_type

    def to_json(self):
        '''
        The function to_json convert ProjectMeta class object to json format
        :return: ProjectMeta in json format(dict)
        '''
        res = {
            ProjectMetaJsonFields.OBJ_CLASSES: self._obj_classes.to_json(),
            ProjectMetaJsonFields.TAGS: self._tag_metas.to_json()
        }
        if self._project_type is not None:
            res[ProjectMetaJsonFields.PROJECT_TYPE] = self._project_type
        return res

    @classmethod
    def from_json(cls, data):
        '''
        The function from_json convert ProjectMeta from json format to ProjectMeta class object. Generate exception error if all project tags not in a single collection
        :param data: input ProjectMeta in json format
        :return: ProjectMeta class object
        '''
        tag_metas_json = data.get(ProjectMetaJsonFields.TAGS, [])
        img_tag_metas_json = data.get(ProjectMetaJsonFields.IMG_TAGS, [])
        obj_tag_metas_json = data.get(ProjectMetaJsonFields.OBJ_TAGS, [])
        project_type = data.get(ProjectMetaJsonFields.PROJECT_TYPE, None)

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
                   tag_metas=tag_metas, project_type=project_type)

    def merge(self, other):
        '''
        Merge all instances from given meta to ProjectMeta and return it copy
        :param other: ProjectMeta class object
        :return: ProjectMeta class object
        '''
        return self.clone(obj_classes=self._obj_classes.merge(other.obj_classes),
                          tag_metas=self._tag_metas.merge(other._tag_metas))

    def clone(self, obj_classes: ObjClassCollection = None, tag_metas: TagMetaCollection = None, project_type=None):
        '''
        The function clone create copy of ProjectMeta with given Collections that stores ObjClass and TagMeta
        :param obj_classes: ObjClassCollection class object
        :param tag_metas: TagMetaCollection class object
        :return: ProjectMeta class object
        '''
        return ProjectMeta(obj_classes=take_with_default(obj_classes, self.obj_classes),
                           tag_metas=take_with_default(tag_metas, self.tag_metas),
                           project_type=take_with_default(project_type, self.project_type))

    def add_obj_class(self, new_obj_class):
        '''
        The function add_obj_class add given objclass to ProjectMeta collection that stores ObjClass instances and return copy of ProjectMeta
        :param new_obj_class: ObjClass class object
        :return: ProjectMeta class object
        '''
        return self.add_obj_classes([new_obj_class])

    def add_obj_classes(self, new_obj_classes):
        '''
        The function add_obj_class add given objclasses to ProjectMeta collection that stores ObjClass instances and return copy of ProjectMeta
        :param new_obj_classes: list of ObjClass class objects
        :return: ProjectMeta class object
        '''
        return self.clone(obj_classes=self.obj_classes.add_items(new_obj_classes))

    def add_tag_meta(self, new_tag_meta):
        '''
        The function add_tag_meta add given tag to ProjectMeta collection that stores TagMeta instances and return copy of ProjectMeta
        :param new_tag_meta: TagMeta class object
        :return: ProjectMeta class object
        '''
        return self.add_tag_metas([new_tag_meta])

    def add_tag_metas(self, new_tag_metas):
        '''
        The function add_tag_metas add given tags to ProjectMeta collection that stores TagMeta instances and return copy of ProjectMeta
        :param new_tag_metas: list of TagMeta class objects
        :return: ProjectMeta class object
        '''
        return self.clone(tag_metas=self.tag_metas.add_items(new_tag_metas))

    @staticmethod
    def _delete_items(collection, item_names):
        '''
        :param collection: ObjClassCollection or TagMetaCollection instance
        :param item_names: list of item names to delete
        :return: list of items, which are in collection and not in given list of items to delete
        '''
        names_to_delete = set(item_names)
        res_items = []
        for item in collection:
            if item.key() not in names_to_delete:
                res_items.append(item)
        return res_items

    def delete_obj_class(self, obj_class_name):
        '''
        The function delete_obj_class delete objclass with given name from ProjectMeta collection that stores ObjClass instances and return copy of ProjectMeta
        :param obj_class_name: str(name of ObjClass to detele from collection)
        :return: ProjectMeta class object
        '''
        return self.delete_obj_classes([obj_class_name])

    def delete_obj_classes(self, obj_class_names):
        '''
        The function delete_obj_classes delete objclasses with given list of names from ProjectMeta collection that stores ObjClass instances and return copy of ProjectMeta
        :param obj_class_names: list of names ObjClass objects to delete
        :return: ProjectMeta class object
        '''
        res_items = self._delete_items(self._obj_classes, obj_class_names)
        return self.clone(obj_classes=ObjClassCollection(res_items))

    def delete_tag_meta(self, tag_name):
        '''
        The function delete_tag_meta delete tag with given name from ProjectMeta collection that stores TagMeta instances and return copy of ProjectMeta
        :param tag_name: str(name of TagMeta to detele from collection)
        :return: ProjectMeta class object
        '''
        return self.delete_tag_metas([tag_name])

    def delete_tag_metas(self, tag_names):
        '''
        The function delete_tag_metas delete tags with given list of names from ProjectMeta collection that stores TagMeta instances and return copy of ProjectMeta
        :param tag_names: list of names TagMeta objects to delete
        :return: ProjectMeta class object
        '''
        res_items = self._delete_items(self._tag_metas, tag_names)
        return self.clone(tag_metas=TagMetaCollection(res_items))

    def get_obj_class(self, obj_class_name):
        '''
        :param obj_class_name: str
        :return: ObjClass class object with given name from ProjectMeta collection that stores ObjClass instances
        '''
        return self._obj_classes.get(obj_class_name)

    def get_tag_meta(self, tag_name):
        '''
        :param tag_name: str
        :return: TagMeta class object with given name from ProjectMeta collection that stores TagMeta instances
        '''
        return self._tag_metas.get(tag_name)

    @staticmethod
    def merge_list(metas):
        '''
        The function merge_list merge metas from given list of metas in single ProjectMeta class object
        :param metas: list of ProjectMeta objects
        :return: ProjectMeta class object
        '''
        res_meta = ProjectMeta()
        for meta in metas:
            res_meta = res_meta.merge(meta)
        return res_meta

    def __str__(self):
        result = 'ProjectMeta:\n'
        result += 'Object Classes\n{}\n'.format(str(self._obj_classes))
        result += 'Tags\n{}\n'.format(str(self._tag_metas))
        return result

    def __eq__(self, other: ProjectMeta):
        if self.obj_classes == other.obj_classes and self.tag_metas == other.tag_metas:
            return True
        return False

    def __ne__(self, other: ProjectMeta):
        return not self == other

    def to_segmentation_task(self, keep_geometries=[Polygon, Bitmap]) -> (ProjectMeta, dict):
        mapping = {}
        res_classes = []
        for obj_class in self.obj_classes:
            obj_class: ObjClass
            if obj_class.geometry_type in keep_geometries:
                if obj_class.geometry_type == Bitmap:
                    mapping[obj_class] = obj_class
                    res_classes.append(obj_class)
                else:
                    new_class = obj_class.clone(geometry_type=Bitmap)
                    mapping[obj_class] = new_class
                    res_classes.append(new_class)
            else:
                mapping[obj_class] = None
        res_meta = self.clone(obj_classes=ObjClassCollection(res_classes))
        return res_meta, mapping

    def to_detection_task(self, convert_classes=False) -> (ProjectMeta, dict):
        mapping = {}
        res_classes = []
        for obj_class in self.obj_classes:
            obj_class: ObjClass
            if obj_class.geometry_type == Rectangle:
                mapping[obj_class] = obj_class
                res_classes.append(obj_class)
            else:
                if convert_classes is True:
                    new_class = obj_class.clone(geometry_type=Rectangle)
                    mapping[obj_class] = new_class
                    res_classes.append(new_class)
                else:
                    # ignore class
                    mapping[obj_class] = None
        res_meta = self.clone(obj_classes=ObjClassCollection(res_classes))
        return res_meta, mapping


