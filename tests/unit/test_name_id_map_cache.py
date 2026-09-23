# coding: utf-8

"""The name→id maps an ApiContext keeps, and what happens when the project meta moves.

Reading the project's classes and tags once per bulk upload instead of once per item was
worth over a third of an annotation upload, so the maps are cached for the lifetime of an
open `ApiContext`. The cost of that is a map that can go stale: an inference app adds a
class to the meta and uploads annotations using it inside the same context, and the class
it just created is not in the map. It used to be a KeyError on a name the caller had every
right to use.
"""

import pytest

from supervisely.api.entity_annotation.tag_api import TagApi
from supervisely.api.object_class_api import ObjectClassApi


class _Info:
    def __init__(self, name, id):
        self.name = name
        self.id = id


class _FakeApi:
    def __init__(self):
        # What `ApiContext.__enter__` sets. It matters that this is not empty: outside a
        # context the attribute is `{}`, and the maps read that as "no bulk operation is
        # running, do not cache".
        self.optimization_context = {
            "project_id": 1,
            "dataset_id": None,
            "project_meta": None,
            "with_alpha_masks": False,
        }


def _api_with(cls, names):
    """One of the two map APIs, wired to a project whose names the test can change."""
    api = _FakeApi()
    instance = cls.__new__(cls)
    instance._api = api
    reads = []

    def get_list(project_id):
        reads.append(project_id)
        return [_Info(name, id) for id, name in enumerate(names, start=1)]

    instance.get_list = get_list
    return instance, reads


@pytest.mark.parametrize("cls", [ObjectClassApi, TagApi])
def test_the_map_is_read_once_inside_a_context(cls):
    """The reason the cache exists at all."""
    names = ["cat"]
    api, reads = _api_with(cls, names)

    assert api.get_name_to_id_map(1) == {"cat": 1}
    assert api.get_name_to_id_map(1) == {"cat": 1}
    assert reads == [1]


@pytest.mark.parametrize("cls", [ObjectClassApi, TagApi])
def test_a_refresh_sees_what_was_added_after_it_was_cached(cls):
    """Without this the caller is stuck with the map as it was when the context opened."""
    names = ["cat"]
    api, reads = _api_with(cls, names)

    assert api.get_name_to_id_map(1) == {"cat": 1}
    names.append("dog")
    # The stale map is what a plain call still answers with...
    assert api.get_name_to_id_map(1) == {"cat": 1}
    # ...and the refresh is what sees the new name, and replaces the cached map with it.
    assert api.get_name_to_id_map(1, refresh=True) == {"cat": 1, "dog": 2}
    assert api.get_name_to_id_map(1) == {"cat": 1, "dog": 2}
    # Two reads: the first one, and the refresh. The cached calls in between read nothing,
    # which is the whole point of the cache.
    assert reads == [1, 1]


@pytest.mark.parametrize("cls", [ObjectClassApi, TagApi])
def test_without_a_context_nothing_is_cached(cls):
    """Outside a bulk operation there is no window for the map to go stale in."""
    names = ["cat"]
    instance = cls.__new__(cls)
    instance._api = object()  # no optimization_context at all
    reads = []

    def get_list(project_id):
        reads.append(project_id)
        return [_Info(name, id) for id, name in enumerate(names, start=1)]

    instance.get_list = get_list

    instance.get_name_to_id_map(1)
    instance.get_name_to_id_map(1)
    assert reads == [1, 1]
