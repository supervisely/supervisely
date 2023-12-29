# coding: utf-8

from typing import Dict, List, Optional, Union

from jsonschema import ValidationError, validate

from supervisely.collection.key_indexed_collection import KeyObject
from supervisely.io.json import JsonSerializable


class ProjectSettingsJsonFields:
    MULTI_VIEW = "multiView"
    ENABLED = "enabled"
    TAG_ID = "tagId"
    TAG_NAME = "tagName"
    IS_SYNCED = "areSynced"


class ProjectSettingsRequiredSchema:
    SCHEMA = {
        "type": "object",
        "properties": {
            ProjectSettingsJsonFields.MULTI_VIEW: {
                "type": "object",
                "properties": {
                    ProjectSettingsJsonFields.ENABLED: {"type": "boolean"},
                    ProjectSettingsJsonFields.TAG_ID: {"type": "integer"},
                    ProjectSettingsJsonFields.TAG_NAME: {"type": "string"},
                    ProjectSettingsJsonFields.IS_SYNCED: {"type": "boolean"},
                },
                "required": [
                    ProjectSettingsJsonFields.ENABLED,
                    ProjectSettingsJsonFields.TAG_ID,
                    ProjectSettingsJsonFields.TAG_NAME,
                    ProjectSettingsJsonFields.IS_SYNCED,
                ],
                "additionalProperties": False,
            }
        },
        "required": [ProjectSettingsJsonFields.MULTI_VIEW],
        "additionalProperties": False,
    }


def validate_settings_json(data: dict) -> None:
    try:
        validate(instance=data, schema=ProjectSettingsRequiredSchema.SCHEMA)
    except ValidationError as e:
        msg = f"The validation of the field {e.json_path} has failed with the following message: {e.message}."
        if e.validator == "required":
            raise ValidationError(
                f"{msg} Check the correctness of the field names in the {ProjectSettingsJsonFields}."
            )
        elif e.validator == "type":
            raise ValidationError(
                f"{msg} Check the correctness of the field value types in the {ProjectSettingsRequiredSchema}."
            )
        else:
            raise ValidationError(msg)


class ProjectSettings(JsonSerializable):
    def __init__(
        self,
        multiview_enabled: bool = False,
        multiview_tag_name: str = None,
        multiview_tag_id: int = None,
        multiview_is_synced: bool = False,
    ):
        self.multiview_enabled = multiview_enabled
        self.multiview_tag_name = multiview_tag_name
        self.multiview_tag_id = multiview_tag_id
        self.multiview_is_synced = multiview_is_synced

        if multiview_enabled is True:
            if multiview_tag_name is None and multiview_tag_id is None:
                raise ValueError(
                    "When multi-view mode is enabled, the value of multiview_tag_name or multiview_tag_id should be defined."
                )
            else:
                if multiview_is_synced is None:
                    raise ValueError(
                        "When multi-view mode is enabled, the value of views_are_synced should be defined."
                    )

    def to_json(self) -> dict:
        return {
            ProjectSettingsJsonFields.MULTI_VIEW: {
                ProjectSettingsJsonFields.ENABLED: self.multiview_enabled,
                ProjectSettingsJsonFields.TAG_NAME: self.multiview_tag_name,
                ProjectSettingsJsonFields.TAG_ID: self.multiview_tag_id,
                ProjectSettingsJsonFields.IS_SYNCED: self.multiview_is_synced,
            }
        }

    @classmethod
    def from_json(cls, data: Dict) -> "ProjectSettings":
        validate_settings_json(data)

        d_multiview = data[ProjectSettingsJsonFields.MULTI_VIEW]

        return cls(
            multiview_enabled=d_multiview[ProjectSettingsJsonFields.ENABLED],
            multiview_tag_name=d_multiview[ProjectSettingsJsonFields.TAG_NAME],
            multiview_tag_id=d_multiview[ProjectSettingsJsonFields.TAG_ID],
            multiview_is_synced=d_multiview[ProjectSettingsJsonFields.IS_SYNCED],
        )
