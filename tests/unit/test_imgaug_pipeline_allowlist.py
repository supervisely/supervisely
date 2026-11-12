"""build_pipeline must only build imgaug augmenters, never other attributes reachable from iaa.

imgaug.augmenters re-exports modules such as os and sys, so a pipeline config entry
{"category": "os", "name": "system", "params": {"command": ...}} used to run a shell command.
A fake imgaug package is used because the real one does not import with NumPy 2.
"""

import sys
import types

import pytest


@pytest.fixture
def fake_imgaug(monkeypatch):
    imgaug = types.ModuleType("imgaug")
    iaa = types.ModuleType("imgaug.augmenters")
    meta = types.ModuleType("imgaug.augmenters.meta")
    arithmetic = types.ModuleType("imgaug.augmenters.arithmetic")

    class Augmenter:
        def __init__(self, **params):
            self.params = params

    class Add(Augmenter):
        pass

    class Sequential(Augmenter):
        def __init__(self, children, random_order=False):
            super().__init__()
            self.children = children

    class Sometimes(Augmenter):
        def __init__(self, p, then_list):
            super().__init__()
            self.then_list = then_list

    def not_an_augmenter(**params):
        raise AssertionError("must not be called")

    meta.Augmenter, meta.Sequential, meta.Sometimes = Augmenter, Sequential, Sometimes
    arithmetic.Add, arithmetic.helper = Add, not_an_augmenter
    iaa.meta, iaa.arithmetic, iaa.Sequential = meta, arithmetic, Sequential
    calls = []
    leaked_os = types.ModuleType("os")
    leaked_os.system = lambda command: calls.append(command)
    iaa.os = leaked_os  # what `from imgaug.augmenters.debug import *` leaks in imgaug 0.4.0
    imgaug.augmenters = iaa
    for name, mod in {
        "imgaug": imgaug,
        "imgaug.augmenters": iaa,
        "imgaug.augmenters.meta": meta,
        "imgaug.augmenters.arithmetic": arithmetic,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return types.SimpleNamespace(iaa=iaa, Add=Add, calls=calls)


def test_augmenter_classes_still_build(fake_imgaug):
    from supervisely.aug import imgaug_utils

    pipeline = imgaug_utils.build_pipeline(
        [{"category": "arithmetic", "name": "Add", "params": {"value": [-10, 10]}}]
    )
    (aug,) = pipeline.children
    assert isinstance(aug, fake_imgaug.Add)
    assert aug.params == {"value": (-10, 10)}


@pytest.mark.parametrize(
    "category,name",
    [("os", "system"), ("arithmetic", "helper"), ("meta", "__class__"), ("nope", "Add")],
)
def test_anything_else_is_rejected_without_being_called(fake_imgaug, category, name):
    from supervisely.aug import imgaug_utils

    with pytest.raises(ValueError, match="Unknown augmentation"):
        imgaug_utils.build_pipeline(
            [{"category": category, "name": name, "params": {"command": "touch /tmp/pwned"}}]
        )
    assert fake_imgaug.calls == []
