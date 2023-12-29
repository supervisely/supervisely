# coding: utf-8


from jsonschema import ValidationError, validate


class ProjectMeta:
    # see <class 'supervisely.project.project_meta.ProjectMeta'>
    pass


class ProjectSettingsJsonFields:
    MULTI_VIEW = "multiView"
    ENABLED = "enabled"
    TAG_ID = "tagId"
    TAG_NAME = "tagName"
    VIEWS_ARE_SYNCED = "viewsAreSynced"


class RequiredSchemaProjectSettings:
    SCHEMA = {
        "type": "object",
        "properties": {
            ProjectSettingsJsonFields.MULTI_VIEW: {
                "type": "object",
                "properties": {
                    ProjectSettingsJsonFields.ENABLED: {"type": "boolean"},
                    ProjectSettingsJsonFields.TAG_ID: {"type": "integer"},
                    ProjectSettingsJsonFields.TAG_NAME: {"type": "string"},
                    ProjectSettingsJsonFields.VIEWS_ARE_SYNCED: {"type": "boolean"},
                },
                "required": [
                    ProjectSettingsJsonFields.ENABLED,
                    ProjectSettingsJsonFields.TAG_ID,
                    ProjectSettingsJsonFields.TAG_NAME,
                    ProjectSettingsJsonFields.VIEWS_ARE_SYNCED,
                ],
                "additionalProperties": False,
            }
        },
        "required": [ProjectSettingsJsonFields.MULTI_VIEW],
        "additionalProperties": False,
    }


def validate_settings_json(data: dict) -> None:
    try:
        validate(instance=data, schema=RequiredSchemaProjectSettings.SCHEMA)
    except ValidationError as e:
        msg = f"The validation of the field {e.json_path} has failed with the following message: {e.message}."
        if e.validator == "required":
            raise ValidationError(
                f"{msg} Check the correctness of the field names in the {ProjectSettingsJsonFields}."
            )
        elif e.validator == "type":
            raise ValidationError(
                f"{msg} Check the correctness of the field value types in the {RequiredSchemaProjectSettings}."
            )
        else:
            raise ValidationError(msg)


class ProjectSettings:
    def __init__(
        self,
        enable_multiview: bool = False,
        multiview_tag_name: str = None,
        multiview_tag_id: int = None,
        multiviews_are_synced: bool = False,
    ):
        self.enable_multiview = enable_multiview
        self.multiview_tag_name = multiview_tag_name
        self.multiview_tag_id = multiview_tag_id
        self.views_are_synced = multiviews_are_synced

        if enable_multiview is True:
            if multiview_tag_name is None and multiview_tag_id is None:
                raise ValueError(
                    "When multi-view mode is enabled, the value of multiview_tag_name or multiview_tag_id should be defined."
                )
            else:
                if multiviews_are_synced is None:
                    raise ValueError(
                        "When multi-view mode is enabled, the value of views_are_synced should be defined."
                    )

    def to_json(self) -> dict:
        return {
            ProjectSettingsJsonFields.MULTI_VIEW: {
                ProjectSettingsJsonFields.ENABLED: self.enable_multiview,
                ProjectSettingsJsonFields.TAG_NAME: self.multiview_tag_name,
                ProjectSettingsJsonFields.TAG_ID: self.multiview_tag_id,
                ProjectSettingsJsonFields.VIEWS_ARE_SYNCED: self.views_are_synced,
            }
        }

    # def from_json(self, json: dict) -> ProjectSettings:
    #     validate_settings_json(json)

    #     json_m = json[ProjectSettingsJsonFields.MULTI_VIEW]

    #     return ProjectSettings(
    #         enable_multiview=json_m[ProjectSettingsJsonFields.ENABLED],
    #         multiview_tag_name=json_m[ProjectSettingsJsonFields.TAG_NAME],
    #         multiview_tag_id=json_m[ProjectSettingsJsonFields.TAG_ID],
    #         multiviews_are_synced=json_m[ProjectSettingsJsonFields.VIEWS_ARE_SYNCED],
    #     )

    def from_json(self, json: dict, project_meta: ProjectMeta = None) -> ProjectMeta:
        validate_settings_json(json)
        json_m = json[ProjectSettingsJsonFields.MULTI_VIEW]

        s = self.clone(
            enable_multiview=json_m[ProjectSettingsJsonFields.ENABLED],
            multiview_tag_name=json_m[ProjectSettingsJsonFields.TAG_NAME],
            multiview_tag_id=json_m[ProjectSettingsJsonFields.TAG_ID],
            multiviews_are_synced=json_m[ProjectSettingsJsonFields.VIEWS_ARE_SYNCED],
        )

        from supervisely.project.project_meta import ProjectMeta

        if isinstance(project_meta, ProjectMeta):
            return ProjectMeta(
                obj_classes=project_meta.obj_classes,
                tag_metas=project_meta.tag_metas,
                project_type=project_meta.project_type,
                project_settings=s,
            )
        return ProjectMeta(
            project_settings=s,
        )

    def clone(
        self,
        enable_multiview: bool = False,
        multiview_tag_name: str = None,
        multiview_tag_id: int = None,
        multiviews_are_synced: bool = False,
    ):
        return ProjectSettings(
            enable_multiview,
            multiview_tag_name,
            multiview_tag_id,
            multiviews_are_synced,
        )
