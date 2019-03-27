# coding: utf-8

from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.annotation.tag import Tag


class TagCollection(KeyIndexedCollection):
    item_type = Tag

    def to_json(self):
        return [tag.to_json() for tag in self]

    @classmethod
    def from_json(cls, data, tag_meta_collection):
        tags = [Tag.from_json(tag_json, tag_meta_collection) for tag_json in data]
        return cls(tags)

    def __str__(self):
        return 'Tags:\n' + super(TagCollection, self).__str__()

