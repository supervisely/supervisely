# coding: utf-8

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
            raise ValidationError(f"The validation failed with the following message: {e.message}.")
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
        multiview_tag_name: Optional[str] = None,
        multiview_tag_id: Optional[int] = None,
        multiview_is_synced: bool = False,
    ):
        self.multiview_enabled = multiview_enabled
        self._multiview_tag_name = multiview_tag_name
        self._multiview_tag_id = multiview_tag_id
        self.multiview_is_synced = multiview_is_synced

    @property
    def multiview_tag_name(self) -> str:
        """
        Name.

        :return: Name
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            class_lemon = sly.ObjClass('lemon', sly.Rectangle)
            print(class_lemon.name)
            # Output: 'lemon'
        """
        return self._multiview_tag_name

    @multiview_tag_name.setter
    def multiview_tag_name(self, new_value):
        self._multiview_tag_name = new_value

    @property
    def multiview_tag_id(self) -> str:
        """
        Name.

        :return: Name
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            class_lemon = sly.ObjClass('lemon', sly.Rectangle)
            print(class_lemon.name)
            # Output: 'lemon'
        """
        return self._multiview_tag_id

    @multiview_tag_id.setter
    def multiview_tag_id(self, new_value):
        self._multiview_tag_id = new_value

    def to_json(self) -> dict:
        data = {
            ProjectSettingsJsonFields.MULTI_VIEW: {
                ProjectSettingsJsonFields.ENABLED: self.multiview_enabled,
                ProjectSettingsJsonFields.TAG_NAME: self._multiview_tag_name,
                ProjectSettingsJsonFields.TAG_ID: self._multiview_tag_id,
                ProjectSettingsJsonFields.IS_SYNCED: self.multiview_is_synced,
            }
        }
        validate_settings_schema(data)
        return data

    @classmethod
    def from_json(cls, data: Dict) -> "ProjectSettings":
        validate_settings_schema(data)

        d_multiview = data[ProjectSettingsJsonFields.MULTI_VIEW]

        return cls(
            multiview_enabled=d_multiview[ProjectSettingsJsonFields.ENABLED],
            multiview_tag_name=d_multiview[ProjectSettingsJsonFields.TAG_NAME],
            multiview_tag_id=d_multiview[ProjectSettingsJsonFields.TAG_ID],
            multiview_is_synced=d_multiview[ProjectSettingsJsonFields.IS_SYNCED],
        )
