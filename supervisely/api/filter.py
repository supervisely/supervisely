# coding: utf-8
"""Helpers for building Supervisely API ``filter`` payloads."""

from __future__ import annotations

from copy import deepcopy
import keyword
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Union

from supervisely.api.module_api import ApiField


ApiFilterJson = List[Dict[str, Any]]
ApiFilterLike = Union[
    None,
    Dict[str, Any],
    "ApiFilterCondition",
    "ApiFilter",
    Iterable[Union[Dict[str, Any], "ApiFilterCondition"]],
]


__all__ = [
    "ApiFilter",
    "ApiFilterCondition",
    "ApiFilterJson",
    "ApiFilterLike",
    "FilterCatalog",
    "FilterField",
    "FilterSet",
    "filters",
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

    The class serializes to the API wire format: a list of condition
    dictionaries with ``field``, ``operator`` and ``value`` keys.
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
        """Convert supported filter inputs into a fresh JSON list."""
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


class FilterField:
    """Endpoint-specific field descriptor with operator helper methods."""

    __slots__ = ("name", "field")

    def __init__(self, field: str, name: Optional[str] = None):
        _validate_field(field)
        if name is not None:
            _validate_attr_name(name)
        object.__setattr__(self, "name", name or field)
        object.__setattr__(self, "field", field)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} objects are immutable")

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
        if self.name == self.field:
            return f"{type(self).__name__}({self.field!r})"
        return f"{type(self).__name__}({self.name!r}, field={self.field!r})"


class FilterSet:
    """Named collection of fields supported by one API endpoint."""

    __slots__ = ("_name", "_fields")

    def __init__(self, name: str, fields: Mapping[str, str]):
        _validate_attr_name(name)

        field_objects = {}
        for attr_name, server_field in fields.items():
            _validate_attr_name(attr_name)
            field_objects[attr_name] = FilterField(server_field, attr_name)

        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_fields", MappingProxyType(field_objects))

    @property
    def resource(self) -> str:
        """Resource name for this filter set."""
        return self._name

    def fields(self) -> List[str]:
        """Return the supported Python field names."""
        return list(self._fields.keys())

    def server_fields(self) -> Dict[str, str]:
        """Return a copy of the Python-name to API-field mapping."""
        return {name: field.field for name, field in self._fields.items()}

    def get(self, name: str) -> FilterField:
        """Return a field by name, raising ``KeyError`` if it is not supported."""
        return self._fields[name]

    def __getitem__(self, name: str) -> FilterField:
        return self.get(name)

    def __getattr__(self, name: str) -> FilterField:
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(
                f"{self._name!r} filters do not include field {name!r}"
            )

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def __iter__(self) -> Iterator[FilterField]:
        return iter(self._fields.values())

    def __len__(self) -> int:
        return len(self._fields)

    def __dir__(self) -> List[str]:
        return sorted(set(super().__dir__()) | set(self._fields.keys()))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} objects are immutable")

    def __repr__(self) -> str:
        fields = ", ".join(self._fields.keys())
        return f"{type(self).__name__}({self._name!r}, fields=[{fields}])"


class FilterCatalog:
    """Catalog of endpoint-specific API filter fields."""

    __slots__ = ("_sets",)

    def __init__(self, specs: Mapping[str, Mapping[str, str]]):
        sets = {}
        for name, fields in specs.items():
            _validate_attr_name(name)
            sets[name] = FilterSet(name, fields)
        object.__setattr__(self, "_sets", MappingProxyType(sets))

    def names(self) -> List[str]:
        """Return available filter set names."""
        return list(self._sets.keys())

    def get(self, name: str) -> FilterSet:
        """Return a filter set by name, raising ``KeyError`` if it is unknown."""
        return self._sets[name]

    def __getitem__(self, name: str) -> FilterSet:
        return self.get(name)

    def __getattr__(self, name: str) -> FilterSet:
        try:
            return self._sets[name]
        except KeyError:
            raise AttributeError(f"API filters do not include resource {name!r}")

    def __contains__(self, name: object) -> bool:
        return name in self._sets

    def __iter__(self) -> Iterator[FilterSet]:
        return iter(self._sets.values())

    def __len__(self) -> int:
        return len(self._sets)

    def __dir__(self) -> List[str]:
        return sorted(set(super().__dir__()) | set(self._sets.keys()))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} objects are immutable")

    def __repr__(self) -> str:
        names = ", ".join(self._sets.keys())
        return f"{type(self).__name__}([{names}])"


def _validate_field(field: str) -> None:
    if not isinstance(field, str) or len(field) == 0:
        raise ValueError("filter field must be a non-empty string")


def _validate_attr_name(name: str) -> None:
    if not isinstance(name, str) or len(name) == 0:
        raise ValueError("filter names must be non-empty strings")
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"filter name {name!r} is not a valid Python attribute")


def _validate_operator(operator: str) -> None:
    if operator not in ApiFilter.VALID_OPERATORS:
        allowed = ", ".join(sorted(ApiFilter.VALID_OPERATORS))
        raise ValueError(
            f"unsupported filter operator: {operator!r}. Allowed operators: {allowed}"
        )


def _snake_to_camel(name: str) -> str:
    chunks = name.split("_")
    return chunks[0] + "".join(chunk[:1].upper() + chunk[1:] for chunk in chunks[1:])


_FIELD_ALIASES = {
    "id": ApiField.ID,
    "name": ApiField.NAME,
    "description": ApiField.DESCRIPTION,
    "created_at": ApiField.CREATED_AT,
    "updated_at": ApiField.UPDATED_AT,
    "dataset_id": ApiField.DATASET_ID,
    "project_id": ApiField.PROJECT_ID,
    "workspace_id": ApiField.WORKSPACE_ID,
    "team_id": ApiField.TEAM_ID,
    "image_id": ApiField.IMAGE_ID,
    "parent_id": ApiField.PARENT_ID,
    "class_id": ApiField.CLASS_ID,
    "entity_id": ApiField.ENTITY_ID,
    "object_id": ApiField.OBJECT_ID,
    "geometry_type": ApiField.GEOMETRY_TYPE,
    "created_by_id": ApiField.CREATED_BY_ID[0][0],
    "assigned_to_id": ApiField.ASSIGNED_TO_ID[0][0],
    "reviewer_id": ApiField.REVIEWER_ID,
    "labeling_queue_id": ApiField.LABELING_QUEUE_ID,
    "labeling_exam_id": ApiField.LABELING_EXAM_ID,
    "frame_index": ApiField.FRAME,
}


def _server_field(name: str) -> str:
    return _FIELD_ALIASES.get(name, _snake_to_camel(name))


def _fields(*names: str, **overrides: str) -> Dict[str, str]:
    fields = {name: _server_field(name) for name in names}
    fields.update(overrides)
    return fields


_TIMESTAMPS = ("created_at", "updated_at")
_NAMED = ("id", "name") + _TIMESTAMPS
_DATASET_ENTITY = ("dataset_id", "project_id")
_DESCRIBED_ENTITY = _NAMED + _DATASET_ENTITY + ("description", "size")


_FILTER_SPECS = {
    "image": _fields(
        *_NAMED,
        *_DATASET_ENTITY,
        "width",
        "height",
        "labels_count",
        "hash",
        "mime",
        "ext",
        "size",
    ),
    "project": _fields(
        *_NAMED,
        "workspace_id",
        "type",
        "size",
        "items_count",
        "datasets_count",
        team_id=ApiField.GROUP_ID,
    ),
    "dataset": _fields(
        *_NAMED,
        "project_id",
        "workspace_id",
        "parent_id",
        "images_count",
        "items_count",
        "size",
        team_id=ApiField.GROUP_ID,
    ),
    "annotation": _fields(*_NAMED, "image_id", "dataset_id"),
    "video": _fields(*_DESCRIBED_ENTITY, "frames_count"),
    "volume": _fields(*_DESCRIBED_ENTITY),
    "pointcloud": _fields(*_DESCRIBED_ENTITY),
    "figure": _fields(
        "id",
        "class_id",
        "entity_id",
        "object_id",
        "project_id",
        "dataset_id",
        "frame_index",
        "geometry_type",
        *_TIMESTAMPS,
    ),
    "object": _fields(
        "id",
        "class_id",
        "entity_id",
        "dataset_id",
        "description",
        "created_by_id",
        *_TIMESTAMPS,
    ),
    "tag": _fields(*_NAMED, "project_id", "color"),
    "team": _fields(*_NAMED, "role"),
    "workspace": _fields(*_NAMED, "team_id", "description"),
    "user": _fields(
        "id",
        "login",
        "name",
        "email",
        "role",
        "role_id",
        "disabled",
        "last_login",
        *_TIMESTAMPS,
    ),
    "task": _fields(
        "id",
        "type",
        "status",
        "workspace_id",
        "user_id",
        "started_at",
        "finished_at",
        *_TIMESTAMPS,
    ),
    "agent": _fields(*_NAMED, "team_id", "status", "version", "type"),
    "plugin": _fields(*_NAMED, "team_id", "description", "type"),
    "role": _fields(*_NAMED, "description"),
    "guide": _fields(*_NAMED, "team_id", "created_by_id"),
    "webhook": _fields(*_NAMED, "team_id", "url", "events"),
    "entity_collection": _fields(*_NAMED, "team_id", "project_id", "type"),
    "labeling_job": _fields(
        *_NAMED,
        "team_id",
        "workspace_id",
        "project_id",
        "dataset_id",
        "created_by_id",
        "assigned_to_id",
        "reviewer_id",
        "status",
        "labeling_queue_id",
        "labeling_exam_id",
    ),
    "labeling_queue_entity": _fields(*_NAMED, "status", "dataset_id", "project_id"),
    "project_version": _fields("id", "project_id", "version", *_TIMESTAMPS),
    "app": _fields("id", "name", "module_id", "slug", "type", "status", *_TIMESTAMPS),
    "object_class": _fields(*_NAMED, "project_id", "geometry_type", "color"),
}


filters = FilterCatalog(_FILTER_SPECS)
