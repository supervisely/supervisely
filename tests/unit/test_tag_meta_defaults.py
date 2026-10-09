import pickle

import numpy as np
import pytest

from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_meta import (
    TagApplicableTo,
    TagMeta,
    TagMetaJsonFields,
    TagValueType,
)
from supervisely.api.entity_annotation.tag_api import TagApi
from supervisely.geometry.rectangle import Rectangle
from supervisely.io.pickle_compat import restore_legacy_defaults
from supervisely.project.project_meta import ProjectMeta

SUBTYPES = ["sedan", "truck"]


def _default_subtype(**kwargs):
    params = dict(
        name="subtype",
        value_type=TagValueType.ONEOF_STRING,
        possible_values=SUBTYPES,
        color=[10, 20, 30],
        applicable_to=TagApplicableTo.OBJECTS_ONLY,
        applicable_classes=["car"],
        is_default=True,
        default_value="sedan",
    )
    params.update(kwargs)
    return TagMeta(**params)


def test_defaults_are_unset_by_default():
    tag_meta = TagMeta("plain", TagValueType.ANY_STRING)

    assert tag_meta.is_default is False
    assert tag_meta.default_value is None


def test_json_without_defaults_has_no_default_keys():
    tag_meta = TagMeta("plain", TagValueType.ANY_STRING, color=[10, 20, 30])

    tag_json = tag_meta.to_json()

    assert TagMetaJsonFields.DEFAULT not in tag_json
    assert TagMetaJsonFields.DEFAULT_VALUE not in tag_json
    restored = TagMeta.from_json(tag_json)
    assert restored.is_default is False
    assert restored.default_value is None


def test_json_round_trip_with_defaults():
    tag_meta = _default_subtype()

    tag_json = tag_meta.to_json()

    assert tag_json["default"] is True
    assert tag_json["default_value"] == "sedan"
    restored = TagMeta.from_json(tag_json)
    assert restored.is_default is True
    assert restored.default_value == "sedan"
    assert restored.to_json() == tag_json


@pytest.mark.parametrize(
    "value_type, default_value",
    [
        (TagValueType.ANY_NUMBER, 0),
        (TagValueType.ANY_NUMBER, 2.5),
        (TagValueType.ANY_STRING, "unknown"),
    ],
)
def test_json_round_trip_default_value_only(value_type, default_value):
    tag_meta = TagMeta("value", value_type, default_value=default_value)

    tag_json = tag_meta.to_json()

    assert TagMetaJsonFields.DEFAULT not in tag_json
    assert tag_json["default_value"] == default_value
    assert TagMeta.from_json(tag_json).default_value == default_value


def test_from_json_reads_server_flat_keys():
    tag_meta = TagMeta.from_json(
        {
            "id": 7,
            "name": "subtype",
            "value_type": "any_string",
            "color": "#0A141E",
            "applicable_type": "objectsOnly",
            "classes": ["car"],
            "default": True,
            "default_value": "sedan",
        }
    )

    assert tag_meta.is_default is True
    assert tag_meta.default_value == "sedan"


def test_from_json_keeps_explicit_false_and_null_as_clear_requests():
    tag_json = {"name": "speed", "value_type": "any_number", "color": "#0A141E"}
    tag_json.update({"default": False, "default_value": None})

    tag_meta = TagMeta.from_json(tag_json)

    assert tag_meta.is_default is False
    assert tag_meta.default_value is None
    assert tag_meta.to_json()["default"] is False
    assert tag_meta.to_json()["default_value"] is None
    assert TagMeta.from_json(tag_meta.to_json()).to_json() == tag_meta.to_json()


# projects.meta.update keeps the stored value for a missing key and for "default": null.
@pytest.mark.parametrize("extra", [{}, {"default": None}], ids=["missing", "null"])
def test_from_json_leaves_unset_defaults_out(extra):
    tag_json = {"name": "speed", "value_type": "any_number", "color": "#0A141E", **extra}

    tag_meta = TagMeta.from_json(tag_json)

    assert TagMetaJsonFields.DEFAULT not in tag_meta.to_json()
    assert TagMetaJsonFields.DEFAULT_VALUE not in tag_meta.to_json()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(applicable_to=TagApplicableTo.ALL),
        dict(applicable_to=TagApplicableTo.IMAGES_ONLY, applicable_classes=[]),
        dict(applicable_classes=[]),
    ],
)
def test_is_default_requires_objects_only_with_classes(kwargs):
    with pytest.raises(ValueError, match="can't be default"):
        _default_subtype(**kwargs)


def test_is_default_must_be_bool():
    with pytest.raises(ValueError, match="is_default"):
        _default_subtype(is_default="yes")


@pytest.mark.parametrize(
    "value_type, possible_values, default_value",
    [
        (TagValueType.NONE, None, "x"),
        (TagValueType.DATE, None, "2026-04-23T15:15:48"),
        (TagValueType.ANY_NUMBER, None, "3"),
        (TagValueType.ANY_NUMBER, None, True),
        (TagValueType.ANY_NUMBER, None, float("nan")),
        (TagValueType.ANY_NUMBER, None, float("inf")),
        (TagValueType.ANY_STRING, None, 3),
        (TagValueType.ANY_STRING, None, "  "),
        (TagValueType.ONEOF_STRING, SUBTYPES, "bus"),
        (TagValueType.ONEOF_STRING, SUBTYPES, 1),
    ],
)
def test_default_value_must_fit_value_type(value_type, possible_values, default_value):
    with pytest.raises(ValueError, match="default_value"):
        TagMeta("value", value_type, possible_values=possible_values, default_value=default_value)


def test_clone_keeps_defaults():
    tag_meta = _default_subtype()

    cloned = tag_meta.clone(name="subtype 2", hotkey="S")

    assert cloned.name == "subtype 2"
    assert cloned.is_default is True
    assert cloned.default_value == "sedan"


def test_clone_replaces_defaults():
    tag_meta = _default_subtype()

    cloned = tag_meta.clone(is_default=False, default_value="truck")

    assert cloned.is_default is False
    assert cloned.default_value == "truck"


def test_clone_validates_kept_default_value():
    tag_meta = _default_subtype()

    with pytest.raises(ValueError, match="default_value"):
        tag_meta.clone(possible_values=["truck", "bus"])


def test_with_default_value_replaces_and_removes():
    tag_meta = _default_subtype()

    assert tag_meta.with_default_value("truck").default_value == "truck"
    removed = tag_meta.with_default_value(None)
    assert removed.default_value is None
    assert removed.is_default is True
    assert removed.to_json()[TagMetaJsonFields.DEFAULT_VALUE] is None
    assert tag_meta.default_value == "sedan"
    with pytest.raises(ValueError, match="default_value"):
        tag_meta.with_default_value("bus")


def test_project_meta_round_trip_keeps_defaults():
    meta = ProjectMeta(obj_classes=[ObjClass("car", Rectangle)], tag_metas=[_default_subtype()])

    restored = ProjectMeta.from_json(meta.to_json())

    tag_meta = restored.get_tag_meta("subtype")
    assert tag_meta.is_default is True
    assert tag_meta.default_value == "sedan"


def test_legacy_pickle_without_default_attributes_restores_unset_defaults():
    tag_meta = TagMeta("plain", TagValueType.ANY_STRING)
    del tag_meta.__dict__["_is_default"]
    del tag_meta.__dict__["_default_value"]

    restored = pickle.loads(pickle.dumps(tag_meta))
    restore_legacy_defaults(restored)

    assert restored.is_default is False
    assert restored.default_value is None


def test_tag_api_bulk_add_payload_includes_defaults():
    tag_api = TagApi.__new__(TagApi)

    settings = tag_api._tag_meta_json(_default_subtype(), {"car": 5})["settings"]

    assert settings["isDefault"] is True
    assert settings["defaultValue"] == "sedan"
    assert settings["classes"] == [5]


def test_tag_api_bulk_add_payload_omits_unset_defaults():
    tag_api = TagApi.__new__(TagApi)

    settings = tag_api._tag_meta_json(TagMeta("plain", TagValueType.ANY_STRING))["settings"]

    assert "isDefault" not in settings
    assert "defaultValue" not in settings


# The server keeps isDefault/defaultValue when the tag's classes are removed later
# (projects.classes.remove) or a one of value is edited in the panel, and still emits them.
@pytest.mark.parametrize(
    "stale",
    [
        {"classes": []},
        {"applicable_type": TagApplicableTo.ALL},
        {"values": ["truck"]},
    ],
)
def test_from_json_drops_stale_defaults_the_server_emits(stale):
    data = _default_subtype().to_json()
    data.update(stale)

    tag_meta = TagMeta.from_json(data)

    if "values" in stale:
        assert tag_meta.is_default is True
        assert tag_meta.default_value is None
    else:
        assert tag_meta.is_default is False
        assert tag_meta.default_value == "sedan"


def test_project_meta_with_stale_default_is_readable():
    meta_json = {
        "classes": [],
        "tags": [
            {
                "name": "subtype",
                "value_type": TagValueType.ONEOF_STRING,
                "values": SUBTYPES,
                "color": "#0A141E",
                "applicable_type": TagApplicableTo.OBJECTS_ONLY,
                "classes": [],
                "default": True,
                "default_value": "sedan",
            }
        ],
    }

    meta = ProjectMeta.from_json(meta_json)

    assert meta.get_tag_meta("subtype").is_default is False
    assert "default" not in meta.to_json()["tags"][0]


def test_constructor_still_rejects_default_without_classes():
    with pytest.raises(ValueError):
        _default_subtype(applicable_classes=[])



def _meta_json_subtype(meta):
    return next(tag for tag in meta.to_json()["tags"] if tag["name"] == "subtype")


# projects.meta.update rejects "default": true for a tag none of whose classes are in the
# meta, so every way of dropping the classes has to leave the flag out of the JSON.
@pytest.mark.parametrize(
    "drop_classes",
    [
        lambda meta: meta.delete_obj_class("car"),
        lambda meta: meta.clone(obj_classes=[ObjClass("truck", Rectangle)]),
        lambda meta: ProjectMeta(obj_classes=[], tag_metas=meta.tag_metas),
    ],
    ids=["delete_obj_class", "clone", "constructor"],
)
def test_meta_json_leaves_out_default_flag_when_its_classes_are_gone(drop_classes):
    meta = ProjectMeta(
        obj_classes=[ObjClass("car", Rectangle), ObjClass("truck", Rectangle)],
        tag_metas=[_default_subtype()],
    )

    tag_json = _meta_json_subtype(drop_classes(meta))

    assert TagMetaJsonFields.DEFAULT not in tag_json
    assert tag_json[TagMetaJsonFields.DEFAULT_VALUE] == "sedan"
    assert tag_json[TagMetaJsonFields.APPLICABLE_CLASSES] == ["car"]


def test_meta_json_keeps_default_flag_while_one_of_its_classes_is_left():
    meta = ProjectMeta(
        obj_classes=[ObjClass("car", Rectangle), ObjClass("truck", Rectangle)],
        tag_metas=[_default_subtype(applicable_classes=["car", "truck"])],
    )

    assert _meta_json_subtype(meta.delete_obj_class("car"))[TagMetaJsonFields.DEFAULT] is True


def test_delete_then_re_add_class_keeps_default_flag():
    meta = ProjectMeta(obj_classes=[ObjClass("car", Rectangle)], tag_metas=[_default_subtype()])

    meta = meta.delete_obj_class("car").add_obj_class(ObjClass("car", Rectangle))

    assert meta.get_tag_meta("subtype").is_default is True
    assert _meta_json_subtype(meta)[TagMetaJsonFields.DEFAULT] is True

def test_string_default_is_stored_trimmed_like_the_server_does():
    tag_meta = TagMeta("note", TagValueType.ANY_STRING, default_value="  checked  ")

    assert tag_meta.default_value == "checked"
    assert tag_meta.with_default_value(" n/a ").default_value == "n/a"
    assert TagMeta.from_json(tag_meta.to_json()).default_value == "checked"


def test_one_of_default_is_not_trimmed():
    with pytest.raises(ValueError):
        _default_subtype(default_value=" sedan ")


@pytest.mark.parametrize(
    "value, expected",
    [(np.int64(2), 2), (np.float32(2.5), 2.5), (np.float64(1.5), 1.5)],
)
def test_numpy_number_default_is_stored_as_a_plain_number(value, expected):
    tag_meta = TagMeta("score", TagValueType.ANY_NUMBER, default_value=value)

    assert tag_meta.default_value == expected
    assert type(tag_meta.default_value) is type(expected)
    assert type(tag_meta.with_default_value(value).default_value) is type(expected)


def test_numpy_bool_is_not_a_number_default():
    with pytest.raises(ValueError):
        TagMeta("score", TagValueType.ANY_NUMBER, default_value=np.bool_(True))


def _json_defaults(tag_meta):
    tag_json = tag_meta.to_json()
    missing = "<missing>"
    return (
        tag_json.get(TagMetaJsonFields.DEFAULT, missing),
        tag_json.get(TagMetaJsonFields.DEFAULT_VALUE, missing),
    )


def test_constructor_writes_explicit_false_and_omits_unset_flag():
    unset = TagMeta("plain", TagValueType.ANY_STRING)
    turned_off = TagMeta("plain", TagValueType.ANY_STRING, is_default=False)

    assert _json_defaults(unset) == ("<missing>", "<missing>")
    assert _json_defaults(turned_off) == (False, "<missing>")


def test_clone_turns_default_off_in_json():
    assert _json_defaults(_default_subtype().clone(is_default=False)) == (False, "sedan")


def test_clone_does_not_turn_an_unset_flag_into_false():
    tag_meta = TagMeta("plain", TagValueType.ANY_STRING)

    assert _json_defaults(tag_meta.clone(name="other")) == ("<missing>", "<missing>")


def test_clear_requests_survive_clone_and_other_copies():
    tag_meta = _default_subtype().clone(is_default=False).with_default_value(None)

    assert _json_defaults(tag_meta) == (False, None)
    assert _json_defaults(tag_meta.clone(hotkey="S")) == (False, None)
    assert _json_defaults(tag_meta.with_frame_range_length_limits(2, 5)) == (False, None)
    assert _json_defaults(tag_meta.add_possible_value("bus")) == (False, None)


def test_setting_a_value_again_drops_the_clear_request():
    tag_meta = _default_subtype().with_default_value(None)

    assert _json_defaults(tag_meta.with_default_value("truck")) == (True, "truck")
    assert _json_defaults(tag_meta.clone(default_value="truck")) == (True, "truck")
    assert _json_defaults(tag_meta.clone(is_default=True)) == (True, None)


def test_stale_default_from_json_is_left_out_not_cleared():
    data = _default_subtype().to_json()
    data.update({"classes": [], "values": ["truck"]})

    tag_meta = TagMeta.from_json(data)

    assert _json_defaults(tag_meta) == ("<missing>", "<missing>")


def test_project_meta_keeps_clear_requests():
    tag_meta = _default_subtype().clone(is_default=False).with_default_value(None)
    meta = ProjectMeta(obj_classes=[ObjClass("car", Rectangle)], tag_metas=[tag_meta])

    restored = ProjectMeta.from_json(meta.to_json())

    assert _meta_json_subtype(restored)[TagMetaJsonFields.DEFAULT] is False
    assert _meta_json_subtype(restored)[TagMetaJsonFields.DEFAULT_VALUE] is None
    # "default": false is accepted for a tag none of whose classes are in the meta
    assert _meta_json_subtype(restored.delete_obj_class("car"))[TagMetaJsonFields.DEFAULT] is False


def test_legacy_pickle_without_clear_attributes_restores_no_clear_requests():
    tag_meta = _default_subtype()
    del tag_meta.__dict__["_clears_default"]
    del tag_meta.__dict__["_clears_default_value"]

    restored = pickle.loads(pickle.dumps(tag_meta))
    restore_legacy_defaults(restored)

    assert _json_defaults(restored) == (True, "sedan")
    assert _json_defaults(restored.clone(hotkey="S")) == (True, "sedan")


def test_tag_api_bulk_add_payload_omits_clear_requests():
    tag_api = TagApi.__new__(TagApi)
    tag_meta = _default_subtype().clone(is_default=False).with_default_value(None)

    settings = tag_api._tag_meta_json(tag_meta, {"car": 5})["settings"]

    assert "isDefault" not in settings
    assert "defaultValue" not in settings


def test_legacy_pickle_of_a_plain_tag_still_leaves_defaults_out():
    tag_meta = TagMeta("plain", TagValueType.ANY_STRING)
    del tag_meta.__dict__["_clears_default"]
    del tag_meta.__dict__["_clears_default_value"]

    restored = pickle.loads(pickle.dumps(tag_meta))
    restore_legacy_defaults(restored)

    assert _json_defaults(restored) == ("<missing>", "<missing>")
    assert _json_defaults(restored.clone(hotkey="P")) == ("<missing>", "<missing>")
