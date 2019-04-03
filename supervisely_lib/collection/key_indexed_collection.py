# coding: utf-8

from prettytable import PrettyTable
from supervisely_lib._utils import take_with_default


class DuplicateKeyError(KeyError):
    pass


class KeyObject:
    def key(self):
        raise NotImplementedError()


class KeyIndexedCollection:
    item_type = KeyObject

    def __init__(self, items=None):
        self._collection = {}
        self._add_items_impl(self._collection, take_with_default(items, []))

    def _add_impl(self, dst_collection, item):
        if not isinstance(item, KeyIndexedCollection.item_type):
            raise TypeError(
                'Item type ({!r}) != {!r}'.format(type(item).__name__, KeyIndexedCollection.item_type.__name__))
        if item.key() in dst_collection:
            raise DuplicateKeyError('Key {!r} already exists'.format(item.key()))
        dst_collection[item.key()] = item

    def _add_items_impl(self, dst_collection, items):
        for item in items:
            self._add_impl(dst_collection, item)

    def add(self, item):
        return self.clone(items=[*self.items(), item])

    def add_items(self, items):
        return self.clone(items=[*self.items(), *items])

    def get(self, key, default=None):
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
        return list(self._collection.values())

    def clone(self, items=None):
        return type(self)(items=(items if items is not None else self.items()))

    def keys(self):
        return list(self._collection.keys())

    def has_key(self, key):
        return key in self._collection

    def intersection(self, other):
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
        items = [item for item in self.items() if item not in other]
        return self.clone(items)

    def merge(self, other):
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