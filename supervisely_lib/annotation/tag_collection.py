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

    @classmethod
    def from_api_response(cls, data, tag_meta_collection, id_to_tagmeta=None):
        if id_to_tagmeta is None:
            id_to_tagmeta = tag_meta_collection.get_id_mapping()
        tags = []
        for tag_json in data:
            tag_meta_id = tag_json["tagId"]
            tag_meta = id_to_tagmeta[tag_meta_id]
            tag_json['name'] = tag_meta.name
            tag = cls.item_type.from_json(tag_json, tag_meta_collection)
            tags.append(tag)
        return cls(tags)
