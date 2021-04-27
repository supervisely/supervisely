# coding: utf-8
from __future__ import annotations
from prettytable import PrettyTable
from supervisely_lib._utils import take_with_default
from typing import List, Iterable
from collections import defaultdict


class DuplicateKeyError(KeyError):
    r"""Raised when trying to add already existing key to
    :class:`KeyIndexedCollection <supervisely_lib.collection.key_indexed_collection.KeyIndexedCollection>`"""
    pass


class KeyObject:
    r"""Base class fo objects that should implement ``key`` method. Child classes then can be stored
    KeyIndexedCollection.
    """
    def key(self):
        r"""
        Returns:
            can be any immutable type; strings and numbers
        """
        raise NotImplementedError()


class KeyIndexedCollection:
    r"""Base class for ObjClassCollection, TagMetaCollection and TagCollection instances
    It is an analogue of python's standard Dict. It allows to store objects inherited from
    :class:`KeyObject <supervisely_lib.collection.key_indexed_collection.KeyObject>`.
    But it raises :class:`DuplicateKeyError <supervisely_lib.collection.key_indexed_collection.DuplicateKeyError>`
    exception when trying to add object with already existing key.
    """
    
    item_type = KeyObject
    """The type of items that can be storred in collection. Defaul value is 
    :class:`KeyObject <supervisely_lib.collection.key_indexed_collection.KeyObject>`. 
    Field has to be overridden in child class. Before adding object to collection its type is compared with 
    ``item_type`` and ``TypeError`` exception is raised if it differs. Collection is immutable.
    """

    def __init__(self, items=None):
        '''
        :param items: dictionary containing collection in format name -> item
        '''
        self._collection = {}
        self._add_items_impl(self._collection, take_with_default(items, []))

    def _add_impl(self, dst_collection, item):
        '''
        Add given item to given collection. Raise error if type of item not KeyObject or item with an item with that name is already in given collection
        :param dst_collection: dictionary
        :param item: ObjClass, TagMeta or Tag class object
        :return: dictionary
        '''
        if not isinstance(item, KeyIndexedCollection.item_type):
            raise TypeError(
                'Item type ({!r}) != {!r}'.format(type(item).__name__, KeyIndexedCollection.item_type.__name__))
        if item.key() in dst_collection:
            raise DuplicateKeyError('Key {!r} already exists'.format(item.key()))
        dst_collection[item.key()] = item

    def _add_items_impl(self, dst_collection, items):
        '''
        Add items from input list to given collection. Raise error if type of item not KeyObject or item with an item with that name is already in given collection
        :param dst_collection: dictionary
        :param items: list of ObjClass, TagMeta or Tag class objects
        '''
        for item in items:
            self._add_impl(dst_collection, item)

    def add(self, item):
        '''
        Add given item to collection
        :param item: ObjClass, TagMeta or Tag class object
        :return: KeyIndexedCollection class object
        '''
        return self.clone(items=[*self.items(), item])

    def add_items(self, items):
        '''
        Add items from given list to collection
        :param items: list of ObjClass, TagMeta or Tag class objects
        :return: KeyIndexedCollection class object
        '''
        """Generators have a ``Yields`` section instead of a ``Returns`` section.

        Args:
            n (int): The upper limit of the range to generate, from 0 to `n` - 1.

        Yields:
            int: The next number in the range of 0 to `n` - 1.

        Examples:
            Examples should be written in doctest format, and should illustrate how
            to use the function.

            >>> print([i for i in example_generator(4)])
            [0, 1, 2, 3]

        """
        return self.clone(items=[*self.items(), *items])

    def get(self, key, default=None):
        '''
        The function get return item from collection with given key(name)
        :param key: str
        :return: ObjClass, TagMeta or Tag class object
        '''
        return self._collection.get(key, default)

    def __next__(self):
        for value in self._collection.values():
            yield value

    def __iter__(self):
        return next(self)

    def __contains__(self, item):
        return (isinstance(item, KeyIndexedCollection.item_type)
                and item == self._collection.get(item.key()))

    def __len__(self):
        return len(self._collection)

    def items(self):
        '''
        :return: list of all items in collection
        '''
        return list(self._collection.values())

    def clone(self, items=None):
        '''
        :param items: list of ObjClass, TagMeta or Tag class objects
        :return: KeyIndexedCollection class object with given items in collection, if given list of items is None - return copy of KeyIndexedCollection
        '''
        return type(self)(items=(items if items is not None else self.items()))

    def keys(self):
        '''
        :return: list of all keys(item names) in collection
        '''
        return list(self._collection.keys())

    def has_key(self, key):
        '''
        Check if given key(item name exist in collection)
        :param key: str
        :return: bool
        '''
        """Exceptions are documented in the same way as classes.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            msg (str): Human readable string describing the exception.
            code (:obj:`int`, optional): Error code.

        Attributes:
            msg (str): Human readable string describing the exception.
            code (int): Exception error code.

        """
        return key in self._collection

    def intersection(self, other):
        '''
        The function intersection find intersection of given list of instances with collection items. Raise error if find items with same keys(item names)
        :param other: list of ObjClass, TagMeta or Tag class objects
        :return: KeyIndexedCollection class object
        '''
        common_items = []
        for other_item in other:
            our_item = self.get(other_item.key())
            if our_item is not None:
                if our_item != other_item:
                    raise ValueError("Different values for the same key {!r}".format(other_item.key()))
                else:
                    common_items.append(our_item)
        return self.clone(common_items)

    def difference(self, other):
        '''
        The function difference find difference between collection and given list of instances
        :param other: list of ObjClass, TagMeta or Tag class objects
        :return: KeyIndexedCollection class object
        '''
        items = [item for item in self.items() if item not in other]
        return self.clone(items)

    def merge(self, other):
        '''
        The function merge merge collection and given list of instances. Raise error if item name from given list is in collection but items in both are different
        :param other: list of ObjClass, TagMeta or Tag class objects
        :return: KeyIndexedCollection class object
        '''
        new_items = []
        for other_item in other.items():
            our_item = self.get(other_item.key())
            if our_item is None:
                new_items.append(other_item)
            elif our_item != other_item:
                raise ValueError('Error during merge for key {!r}: values are different'.format(other_item.key()))
        return self.clone(new_items + self.items())

    def __str__(self):
        res_table = PrettyTable()
        res_table.field_names = self.item_type.get_header_ptable()
        for item in self:
            res_table.add_row(item.get_row_ptable())
        return res_table.get_string()

    def to_json(self) -> List[dict]:
        """
        Converts collection to json serializable list.
        Returns:
            json serializable dictionary
        """
        return [item.to_json() for item in self]

    def __eq__(self, other: KeyIndexedCollection):
        if len(self) != len(other):
            return False
        for cur_item in self:
            other_item = other.get(cur_item.key())
            if other_item is None or cur_item != other_item:
                return False
        return True

    def __ne__(self, other: KeyIndexedCollection):
        return not self == other


class MultiKeyIndexedCollection(KeyIndexedCollection):
    def __init__(self, items=None):
        self._collection = defaultdict(list)
        self._add_items_impl(self._collection, take_with_default(items, []))

    def _add_impl(self, dst_collection, item):
        if not isinstance(item, MultiKeyIndexedCollection.item_type):
            raise TypeError(
                'Item type ({!r}) != {!r}'.format(type(item).__name__, MultiKeyIndexedCollection.item_type.__name__))
        dst_collection[item.key()].append(item)

    def get(self, key, default=None):
        result = self._collection.get(key, default)
        if not result:
            return None
        return result[0]

    def get_all(self, key, default=[]):
        return self._collection.get(key, default)

    def __next__(self):
        for tag_list in self._collection.values():
            for tag in tag_list:
                yield tag

    def __contains__(self, item):
        return (isinstance(item, MultiKeyIndexedCollection.item_type)
                and item in self.get_all(item.key()))

    def __len__(self):
        return sum([len(tag_list) for tag_list in self._collection.values()])

    def items(self):
        res = []
        for tag_list in self._collection.values():
            res.extend(tag_list)
        return res

    def intersection(self, other):
        common_items = []
        for other_item in other:
            key_list = self.get_all(other_item.key())
            for our_item in key_list:
                if our_item == other_item:
                    common_items.append(our_item)
        return self.clone(common_items)

    def merge(self, other):
        new_items = [*self.items(), *other.items()]
        return self.clone(items=new_items)

    def merge_without_duplicates(self, other):
        return super().merge(other)
