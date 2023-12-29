# coding: utf-8

from __future__ import annotations

from typing import Dict, List, Optional, Union

from jsonschema import ValidationError, validate

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
                    ProjectSettingsJsonFields.TAG_ID: {"type": ["integer", "null"]},
                    ProjectSettingsJsonFields.TAG_NAME: {"type": ["string", "null"]},
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


def validate_settings_schema(data: dict) -> None:
    try:
        validate(instance=data, schema=ProjectSettingsRequiredSchema.SCHEMA)
    except ValidationError as e:
        if e.json_path == "$":
            raise ValidationError(
                f"The validation has failed with the following message: {e.message}."
            )
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
    """
    General information about :class:`<supervisely.project.project_settings.ProjectSettings>`. The class is immutable.

    :param multiview_enabled: Enable multi-view mode.
    :type multiview_enabled: bool
    :param multiview_tag_name: The name of the tag which will be used as a group tag for multi-window mode.
    :type multiview_tag_name: str, optional
    :param multiview_tag_id: The id of the tag which will be used as a group tag for multi-window mode.
    :type multiview_tag_id: str, optional
    :param multiview_is_synced: Enable syncronization of views for the multi-view mode.
    :type multiview_is_synced: bool

    :raises: :class:`ValidationError`, if settings schema is corrupted, the exception arises.
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        Example 1: multiView Tag is known (by id or name)
        settings_json = {"multiView": {"enabled": True, "tagName": 'group_tag', "tagId": None, "areSynced": False}}

        Example 2: multiView Tag is unknown, but multiView is enabled. In this case, the tag will be chosen automatically.
        settings_json = {"multiView": {"enabled": True, "tagName": None, "tagId": None, "areSynced": False}}

        settings = sly.ProjectSettings.from_json(settings_json)
    """

    def __init__(
        self,
        multiview_enabled: bool = False,
        multiview_tag_name: Optional[str] = None,
        multiview_tag_id: Optional[int] = None,
        multiview_is_synced: bool = False,
    ):
        self.multiview_enabled = multiview_enabled
        self.multiview_tag_name = multiview_tag_name
        self.multiview_tag_id = multiview_tag_id
        self.multiview_is_synced = multiview_is_synced

    def to_json(self) -> dict:
        data = {
            ProjectSettingsJsonFields.MULTI_VIEW: {
                ProjectSettingsJsonFields.ENABLED: self.multiview_enabled,
                ProjectSettingsJsonFields.TAG_NAME: self.multiview_tag_name,
                ProjectSettingsJsonFields.TAG_ID: self.multiview_tag_id,
                ProjectSettingsJsonFields.IS_SYNCED: self.multiview_is_synced,
            }
        }
        validate_settings_schema(data)
        return data

    @classmethod
    def from_json(cls, data: Dict) -> ProjectSettings:
        validate_settings_schema(data)

        d_multiview = data[ProjectSettingsJsonFields.MULTI_VIEW]

        return cls(
            multiview_enabled=d_multiview[ProjectSettingsJsonFields.ENABLED],
            multiview_tag_name=d_multiview[ProjectSettingsJsonFields.TAG_NAME],
            multiview_tag_id=d_multiview[ProjectSettingsJsonFields.TAG_ID],
            multiview_is_synced=d_multiview[ProjectSettingsJsonFields.IS_SYNCED],
        )
