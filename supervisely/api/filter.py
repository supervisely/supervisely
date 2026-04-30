# coding: utf-8
"""Helpers for building Supervisely API ``filter`` payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Type, Union

from supervisely.api.module_api import ApiField


ApiFilterJson = List[Dict[str, Any]]
ApiFilterLike = Union[
    None,
    Dict[str, Any],
    "ApiFilterCondition",
    "ApiFilter",
    Iterable[Union[Dict[str, Any], "ApiFilterCondition"]],
]


class ApiFilterCondition(dict):
    """Single API filter condition in ``field/operator/value`` format."""

    def __init__(self, field: str, operator: str, value: Any):
        _validate_field(field)
        _validate_operator(operator)
        super().__init__(
            {
                ApiField.FIELD: field,
                ApiField.OPERATOR: operator,
                ApiField.VALUE: value,
            }
        )

    @property
    def field(self) -> str:
        """Server field name used in this condition."""
        return self[ApiField.FIELD]

    @property
    def operator(self) -> str:
        """Server operator used in this condition."""
        return self[ApiField.OPERATOR]

    @property
    def value(self) -> Any:
        """Filter value used in this condition."""
        return self[ApiField.VALUE]

    def to_json(self) -> Dict[str, Any]:
        """Return a fresh JSON-compatible representation."""
        return deepcopy(dict(self))

    def __and__(self, other: ApiFilterLike) -> "ApiFilter":
        return ApiFilter(self) & other


class ApiFilter(list):
    """List-like builder for standard Supervisely API filters.

    The class serializes to the same wire format as the legacy raw list of
    dictionaries, so existing APIs can accept it without changing server
    payloads.
    """

    VALID_OPERATORS = {
        "=",
        "eq",
        "!=",
        "not",
        "in",
        "!in",
        ">",
        "gt",
        ">=",
        "gte",
        "<",
        "lt",
        "<=",
        "lte",
    }

    def __init__(self, filters: ApiFilterLike = None):
        super().__init__(self.normalize(filters))

    @classmethod
    def normalize(cls, filters: ApiFilterLike = None) -> ApiFilterJson:
        """Convert supported filter inputs into a fresh legacy JSON list."""
        if filters is None:
            return []
        if isinstance(filters, ApiFilter):
            return filters.to_json()
        if isinstance(filters, ApiFilterCondition):
            return [filters.to_json()]
        if isinstance(filters, Mapping):
            if len(filters) == 0:
                return []
            return [deepcopy(dict(filters))]
        if hasattr(filters, "to_json") and callable(filters.to_json):
            return cls.normalize(filters.to_json())

        try:
            iterator = iter(filters)
        except TypeError:
            raise TypeError(
                "filters must be None, a dict, ApiFilterCondition, ApiFilter, "
                "or an iterable of dicts/ApiFilterCondition objects"
            )

        result = []
        for item in iterator:
            if isinstance(item, ApiFilter):
                result.extend(item.to_json())
            elif isinstance(item, ApiFilterCondition):
                result.append(item.to_json())
            elif isinstance(item, Mapping):
                result.append(deepcopy(dict(item)))
            elif hasattr(item, "to_json") and callable(item.to_json):
                result.extend(cls.normalize(item.to_json()))
            else:
                raise TypeError(
                    "filter items must be dicts, ApiFilterCondition, or ApiFilter objects"
                )
        return result

    def to_json(self) -> ApiFilterJson:
        """Return a fresh JSON-compatible representation."""
        return deepcopy(list(self))

    def extend(self, filters: ApiFilterLike) -> "ApiFilter":  # type: ignore[override]
        for item in self.normalize(filters):
            super().append(item)
        return self

    def add(self, field: str, operator: str, value: Any) -> "ApiFilter":
        """Append a condition and return this filter for fluent chaining."""
        self.append(ApiFilterCondition(field, operator, value).to_json())
        return self

    def where(self, field: str, operator: str, value: Any) -> "ApiFilter":
        """Alias for :meth:`add`."""
        return self.add(field, operator, value)

    def eq(self, field: str, value: Any) -> "ApiFilter":
        return self.add(field, "=", value)

    def ne(self, field: str, value: Any) -> "ApiFilter":
        return self.add(field, "!=", value)

    def isin(self, field: str, value: Any) -> "ApiFilter":
        return self.add(field, "in", value)

    def in_(self, field: str, value: Any) -> "ApiFilter":
        return self.isin(field, value)

    def notin(self, field: str, value: Any) -> "ApiFilter":
        return self.add(field, "!in", value)

    def not_in(self, field: str, value: Any) -> "ApiFilter":
        return self.notin(field, value)

    def gt(self, field: str, value: Any) -> "ApiFilter":
        return self.add(field, ">", value)

    def gte(self, field: str, value: Any) -> "ApiFilter":
        return self.add(field, ">=", value)

    def lt(self, field: str, value: Any) -> "ApiFilter":
        return self.add(field, "<", value)

    def lte(self, field: str, value: Any) -> "ApiFilter":
        return self.add(field, "<=", value)

    def is_null(self, field: str) -> "ApiFilter":
        return self.eq(field, None)

    def not_null(self, field: str) -> "ApiFilter":
        return self.ne(field, None)

    def between(self, field: str, lower: Any, upper: Any) -> "ApiFilter":
        return self.gte(field, lower).lte(field, upper)

    def __and__(self, other: ApiFilterLike) -> "ApiFilter":
        return ApiFilter(self).extend(other)


class ApiFilterField:
    """Endpoint-specific field descriptor with operator helper methods."""

    def __init__(self, field: str):
        _validate_field(field)
        self.field = field

    def condition(self, operator: str, value: Any) -> ApiFilterCondition:
        return ApiFilterCondition(self.field, operator, value)

    def eq(self, value: Any) -> ApiFilterCondition:
        return self.condition("=", value)

    def ne(self, value: Any) -> ApiFilterCondition:
        return self.condition("!=", value)

    def isin(self, value: Any) -> ApiFilterCondition:
        return self.condition("in", value)

    def in_(self, value: Any) -> ApiFilterCondition:
        return self.isin(value)

    def notin(self, value: Any) -> ApiFilterCondition:
        return self.condition("!in", value)

    def not_in(self, value: Any) -> ApiFilterCondition:
        return self.notin(value)

    def gt(self, value: Any) -> ApiFilterCondition:
        return self.condition(">", value)

    def gte(self, value: Any) -> ApiFilterCondition:
        return self.condition(">=", value)

    def lt(self, value: Any) -> ApiFilterCondition:
        return self.condition("<", value)

    def lte(self, value: Any) -> ApiFilterCondition:
        return self.condition("<=", value)

    def is_null(self) -> ApiFilterCondition:
        return self.eq(None)

    def not_null(self) -> ApiFilterCondition:
        return self.ne(None)

    def between(self, lower: Any, upper: Any) -> ApiFilter:
        return ApiFilter().gte(self.field, lower).lte(self.field, upper)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.field!r})"


def _validate_field(field: str) -> None:
    if not isinstance(field, str) or len(field) == 0:
        raise ValueError("filter field must be a non-empty string")


def _validate_operator(operator: str) -> None:
    if operator not in ApiFilter.VALID_OPERATORS:
        allowed = ", ".join(sorted(ApiFilter.VALID_OPERATORS))
        raise ValueError(f"unsupported filter operator: {operator!r}. Allowed operators: {allowed}")


def _make_filter_builder(name: str, fields: Mapping[str, str]) -> Type:
    attrs = {field_name: ApiFilterField(api_field) for field_name, api_field in fields.items()}
    attrs["__slots__"] = ()
    attrs["__doc__"] = f"Endpoint-specific API filter fields for {name}."
    attrs["__module__"] = __name__
    return type(name, (), attrs)


_COMMON_TIMESTAMPS = {
    "created_at": ApiField.CREATED_AT,
    "updated_at": ApiField.UPDATED_AT,
}

_COMMON_NAMED = {
    "id": ApiField.ID,
    "name": ApiField.NAME,
    **_COMMON_TIMESTAMPS,
}


ImageFilter = _make_filter_builder(
    "ImageFilter",
    {
        **_COMMON_NAMED,
        "dataset_id": ApiField.DATASET_ID,
        "project_id": ApiField.PROJECT_ID,
        "width": ApiField.WIDTH,
        "height": ApiField.HEIGHT,
        "labels_count": ApiField.LABELS_COUNT,
        "hash": ApiField.HASH,
        "mime": ApiField.MIME,
        "ext": ApiField.EXT,
        "size": ApiField.SIZE,
    },
)

ProjectFilter = _make_filter_builder(
    "ProjectFilter",
    {
        **_COMMON_NAMED,
        "workspace_id": ApiField.WORKSPACE_ID,
        "team_id": ApiField.GROUP_ID,
        "type": ApiField.TYPE,
        "size": ApiField.SIZE,
        "items_count": ApiField.ITEMS_COUNT,
        "datasets_count": ApiField.DATASETS_COUNT,
    },
)

DatasetFilter = _make_filter_builder(
    "DatasetFilter",
    {
        **_COMMON_NAMED,
        "project_id": ApiField.PROJECT_ID,
        "workspace_id": ApiField.WORKSPACE_ID,
        "team_id": ApiField.GROUP_ID,
        "parent_id": ApiField.PARENT_ID,
        "images_count": ApiField.IMAGES_COUNT,
        "items_count": ApiField.ITEMS_COUNT,
        "size": ApiField.SIZE,
    },
)

AnnotationFilter = _make_filter_builder(
    "AnnotationFilter",
    {
        **_COMMON_NAMED,
        "image_id": ApiField.IMAGE_ID,
        "dataset_id": ApiField.DATASET_ID,
        "created_at": ApiField.CREATED_AT,
        "updated_at": ApiField.UPDATED_AT,
    },
)

VideoFilter = _make_filter_builder(
    "VideoFilter",
    {
        **_COMMON_NAMED,
        "dataset_id": ApiField.DATASET_ID,
        "project_id": ApiField.PROJECT_ID,
        "description": ApiField.DESCRIPTION,
        "frames_count": "framesCount",
        "size": ApiField.SIZE,
    },
)

VolumeFilter = _make_filter_builder(
    "VolumeFilter",
    {
        **_COMMON_NAMED,
        "dataset_id": ApiField.DATASET_ID,
        "project_id": ApiField.PROJECT_ID,
        "description": ApiField.DESCRIPTION,
        "size": ApiField.SIZE,
    },
)

PointcloudFilter = _make_filter_builder(
    "PointcloudFilter",
    {
        **_COMMON_NAMED,
        "dataset_id": ApiField.DATASET_ID,
        "project_id": ApiField.PROJECT_ID,
        "description": ApiField.DESCRIPTION,
        "size": ApiField.SIZE,
    },
)

FigureFilter = _make_filter_builder(
    "FigureFilter",
    {
        "id": ApiField.ID,
        "class_id": ApiField.CLASS_ID,
        "entity_id": ApiField.ENTITY_ID,
        "object_id": ApiField.OBJECT_ID,
        "project_id": ApiField.PROJECT_ID,
        "dataset_id": ApiField.DATASET_ID,
        "frame_index": ApiField.FRAME,
        "geometry_type": ApiField.GEOMETRY_TYPE,
        **_COMMON_TIMESTAMPS,
    },
)

ObjectFilter = _make_filter_builder(
    "ObjectFilter",
    {
        "id": ApiField.ID,
        "class_id": ApiField.CLASS_ID,
        "entity_id": ApiField.ENTITY_ID,
        "dataset_id": ApiField.DATASET_ID,
        "description": ApiField.DESCRIPTION,
        "created_by_id": ApiField.CREATED_BY_ID[0][0],
        **_COMMON_TIMESTAMPS,
    },
)

TagFilter = _make_filter_builder(
    "TagFilter",
    {
        **_COMMON_NAMED,
        "project_id": ApiField.PROJECT_ID,
        "color": ApiField.COLOR,
    },
)

TeamFilter = _make_filter_builder(
    "TeamFilter",
    {
        **_COMMON_NAMED,
        "role": ApiField.ROLE,
    },
)

WorkspaceFilter = _make_filter_builder(
    "WorkspaceFilter",
    {
        **_COMMON_NAMED,
        "team_id": ApiField.TEAM_ID,
        "description": ApiField.DESCRIPTION,
    },
)

UserFilter = _make_filter_builder(
    "UserFilter",
    {
        "id": ApiField.ID,
        "login": ApiField.LOGIN,
        "name": ApiField.NAME,
        "email": ApiField.EMAIL,
        "role": ApiField.ROLE,
        "role_id": ApiField.ROLE_ID,
        "disabled": ApiField.DISABLED,
        "last_login": ApiField.LAST_LOGIN,
        **_COMMON_TIMESTAMPS,
    },
)

TaskFilter = _make_filter_builder(
    "TaskFilter",
    {
        "id": ApiField.ID,
        "type": ApiField.TYPE,
        "status": ApiField.STATUS,
        "workspace_id": ApiField.WORKSPACE_ID,
        "user_id": ApiField.USER_ID,
        "started_at": ApiField.STARTED_AT,
        "finished_at": ApiField.FINISHED_AT,
        **_COMMON_TIMESTAMPS,
    },
)

AgentFilter = _make_filter_builder(
    "AgentFilter",
    {
        **_COMMON_NAMED,
        "team_id": ApiField.TEAM_ID,
        "status": ApiField.STATUS,
        "version": ApiField.VERSION,
        "type": ApiField.TYPE,
    },
)

PluginFilter = _make_filter_builder(
    "PluginFilter",
    {
        **_COMMON_NAMED,
        "team_id": ApiField.TEAM_ID,
        "description": ApiField.DESCRIPTION,
        "type": ApiField.TYPE,
    },
)

RoleFilter = _make_filter_builder(
    "RoleFilter",
    {
        **_COMMON_NAMED,
        "description": ApiField.DESCRIPTION,
    },
)

GuideFilter = _make_filter_builder(
    "GuideFilter",
    {
        **_COMMON_NAMED,
        "team_id": ApiField.TEAM_ID,
        "created_by_id": ApiField.CREATED_BY_ID[0][0],
    },
)

WebhookFilter = _make_filter_builder(
    "WebhookFilter",
    {
        **_COMMON_NAMED,
        "team_id": ApiField.TEAM_ID,
        "url": ApiField.URL,
        "events": ApiField.EVENTS,
    },
)

EntityCollectionFilter = _make_filter_builder(
    "EntityCollectionFilter",
    {
        **_COMMON_NAMED,
        "team_id": ApiField.TEAM_ID,
        "project_id": ApiField.PROJECT_ID,
        "type": ApiField.TYPE,
    },
)

LabelingJobFilter = _make_filter_builder(
    "LabelingJobFilter",
    {
        **_COMMON_NAMED,
        "team_id": ApiField.TEAM_ID,
        "workspace_id": ApiField.WORKSPACE_ID,
        "project_id": ApiField.PROJECT_ID,
        "dataset_id": ApiField.DATASET_ID,
        "created_by_id": ApiField.CREATED_BY_ID[0][0],
        "assigned_to_id": ApiField.ASSIGNED_TO_ID[0][0],
        "reviewer_id": ApiField.REVIEWER_ID,
        "status": ApiField.STATUS,
        "labeling_queue_id": ApiField.LABELING_QUEUE_ID,
        "labeling_exam_id": ApiField.LABELING_EXAM_ID,
    },
)

LabelingQueueEntityFilter = _make_filter_builder(
    "LabelingQueueEntityFilter",
    {
        **_COMMON_NAMED,
        "id": ApiField.ID,
        "status": ApiField.STATUS,
        "dataset_id": ApiField.DATASET_ID,
        "project_id": ApiField.PROJECT_ID,
    },
)

ProjectVersionFilter = _make_filter_builder(
    "ProjectVersionFilter",
    {
        "id": ApiField.ID,
        "project_id": ApiField.PROJECT_ID,
        "version": ApiField.VERSION,
        "created_at": ApiField.CREATED_AT,
        "updated_at": ApiField.UPDATED_AT,
    },
)

AppFilter = _make_filter_builder(
    "AppFilter",
    {
        "id": ApiField.ID,
        "name": ApiField.NAME,
        "module_id": ApiField.MODULE_ID,
        "slug": ApiField.SLUG,
        "type": ApiField.TYPE,
        "status": ApiField.STATUS,
        "created_at": ApiField.CREATED_AT,
        "updated_at": ApiField.UPDATED_AT,
    },
)

ObjectClassFilter = _make_filter_builder(
    "ObjectClassFilter",
    {
        **_COMMON_NAMED,
        "project_id": ApiField.PROJECT_ID,
        "geometry_type": ApiField.GEOMETRY_TYPE,
        "color": ApiField.COLOR,
    },
)
