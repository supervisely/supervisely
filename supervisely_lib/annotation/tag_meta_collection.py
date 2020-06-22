# coding: utf-8

from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.io.json import JsonSerializable
from supervisely_lib.annotation.tag_meta import TagMeta


class TagMetaCollection(KeyIndexedCollection, JsonSerializable):
    """
    Collection that stores TagMeta instances with unique names.
    """
    item_type = TagMeta

    def to_json(self):
        """
        Converts collection to json serializable list.
        Returns: json serializable dictionary
        """
        return [tag_meta.to_json() for tag_meta in self]

    @classmethod
    def from_json(cls, data):
        """
        Creates collection from json serializable list.
        Returns: TagMetaCollection
        """
        tags = [TagMeta.from_json(tag_meta_json) for tag_meta_json in data]
        return cls(tags)


def make_renamed_tag_metas(src_tag_metas: TagMetaCollection, renamer, skip_missing=False) -> TagMetaCollection:
    '''
    The function make_renamed_classes rename classes names in given collection and return new collection
    :return: TagMetaCollection
    '''
    result_tags = []
    for src_tag in src_tag_metas:
        renamed_name = renamer.rename(src_tag.name)
        if renamed_name is not None:
            result_tags.append(src_tag.clone(name=renamed_name))
        elif not skip_missing:
            raise KeyError(
                'Tag meta named {} could not be mapped to a destination name.'.format(src_tag.name))
    return TagMetaCollection(items=result_tags)

