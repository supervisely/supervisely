import pickle

import pytest

from supervisely.annotation.tag_meta import (
    TagApplicableTo,
    TagMeta,
    TagMetaJsonFields,
    TagValueType,
)
from supervisely.api.entity_annotation.tag_api import TagApi
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


def test_from_json_treats_null_default_value_as_unset():
    tag_meta = TagMeta.from_json(
        {"name": "speed", "value_type": "any_number", "default": False, "default_value": None}
    )

    assert tag_meta.is_default is False
    assert tag_meta.default_value is None


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
    assert TagMetaJsonFields.DEFAULT_VALUE not in removed.to_json()
    assert tag_meta.default_value == "sedan"
    with pytest.raises(ValueError, match="default_value"):
        tag_meta.with_default_value("bus")


def test_project_meta_round_trip_keeps_defaults():
    meta = ProjectMeta(tag_metas=[_default_subtype()])

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
