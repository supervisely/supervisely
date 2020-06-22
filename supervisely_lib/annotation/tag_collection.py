# coding: utf-8

from supervisely_lib.collection.key_indexed_collection import MultiKeyIndexedCollection
from supervisely_lib.annotation.tag import Tag


class TagCollection(MultiKeyIndexedCollection):
    """
    Collection that stores Tag instances with unique names.
    """
    item_type = Tag

    def to_json(self):
        """
        Converts collection to json serializable list.
        Returns: json serializable dictionary
        """
        return [tag.to_json() for tag in self]

    @classmethod
    def from_json(cls, data, tag_meta_collection):
        """
        Creates collection from json serializable list.
        Returns: TagCollection
        """
        tags = [cls.item_type.from_json(tag_json, tag_meta_collection) for tag_json in data]
        return cls(tags)

    def __str__(self):
        return 'Tags:\n' + super(TagCollection, self).__str__()
