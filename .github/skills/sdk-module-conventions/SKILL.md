---
name: sdk-module-conventions
description: >
  Rules for creating new API modules in the Supervisely SDK. Use when adding a new
  *Api class, a new *Info NamedTuple, registering a module on the Api object, or
  when asked to review/fix an existing module for compliance with SDK conventions.
---

# SDK Module Conventions

## File layout

- One module = one file under `supervisely/api/` (or a subdirectory for grouped APIs
  like `video/`, `pointcloud/`, `volume/`).
- File starts with `# coding: utf-8` and a one-line module docstring.
- `from __future__ import annotations` is the first import.

```
# coding: utf-8
"""<One-line description of what this module wraps.>"""

from __future__ import annotations
```

---

## *Info NamedTuple

Every module that represents a server resource must define a `NamedTuple` named
`<Resource>Info`.

Rules:
1. Inherit from `NamedTuple`.
2. All fields have `#:` doc-comments above them.
3. Fields that are always present (id, name) go first **without** default values.
4. All optional / newer fields go after required ones **with** default values (e.g. `= None`).
5. End the class with this guard comment so reviewers know not to remove defaults:

```python
# DO NOT DELETE THIS COMMENT
#! New fields must be added with default values to keep backward compatibility.
```

6. The `id` field is always `int` (not `Optional[int]`).
7. Field names use `snake_case`; server JSON keys are `camelCase` — the mapping is
   declared in `info_sequence()`, not in the NamedTuple itself.

---

## *Api class

Inherit from the most specific applicable base:

| Use case | Base class |
|----------|-----------|
| Standard read + bulk remove | `RemoveableBulkModuleApi` |
| Standard read + single remove | `RemoveableModuleApi` |
| Read-only | `ModuleApiBase` |

### Required overrides

```python
@staticmethod
def info_sequence() -> List[str]:
    """Return the ordered list of server JSON keys that map to *Info fields."""
    return [ApiField.ID, ApiField.NAME, ...]

@staticmethod
def info_tuple_name() -> str:
    """Return the exact string name of the *Info NamedTuple."""
    return "MyResourceInfo"
```

`ModuleApiBase.__init_subclass__` auto-generates `cls.InfoType` from these two methods.

### `_convert_json_info`

- **Do not override** unless the server response requires normalization before the
  generic loop can work (e.g. a field only present inside a nested dict).
- When you do override, call `super()._convert_json_info(normalized, skip_missing=skip_missing)`
  and apply your normalizations to a copy of `info` **before** the call:

```python
def _convert_json_info(self, info: dict, skip_missing=True):
    if info is None:
        return None
    normalized = dict(info)
    # ... patch normalized ...
    return super()._convert_json_info(normalized, skip_missing=skip_missing)
```

- Default `skip_missing=True` (tolerant). Use `skip_missing=False` only when all
  fields are guaranteed to be present (typically internal batch endpoints).

### `_CREATED_BY_FIELD` pattern

When `createdBy` is a nested field (e.g. `ApiField.CREATED_BY_ID = (["createdBy"], "created_by")`),
define a module-level constant so `info_sequence()` and `_convert_json_info` stay in sync:

```python
_CREATED_BY_FIELD = ApiField.CREATED_BY_ID[0][0]  # "createdBy"
```

---

## Registering the module on `Api`

1. Add an import at the top of `supervisely/api/api.py`:
   ```python
   import supervisely.api.my_module_api as my_module_api
   ```
2. Instantiate in `Api.__init__` after related modules:
   ```python
   self.my_module = my_module_api.MyModuleApi(self)
   ```

---

## Exporting public symbols

Add the new `*Api`, `*Info`, and any builder classes to `supervisely/__init__.py`:

```python
from supervisely.api.my_module_api import MyModuleApi, MyModuleInfo
```

---

## Documentation — mandatory rule

**Every new API module must be added to `docs/source/sdk_packages.rst` before the PR
is merged. Existing modules must be updated if their public interface changed.**

### Where to add

There are two places in `sdk_packages.rst`:

1. **Section "API"** (the flat list under the `Api` class) — add all public symbols
   (`*Api`, `*Info`, builder classes):
   ```rst
   ~supervisely.api.my_module_api.MyModuleApi
   ~supervisely.api.my_module_api.MyModuleInfo
   ```

2. **Dedicated section** after the most related type-specific section — add the same
   symbols in a new `autosummary` block:
   ```rst
   My Module API
   -------------
   API for working with <resource> in Supervisely.

   .. autosummary::
       :toctree: sdk
       :nosignatures:
       :template: autosummary/custom-class-template.rst

       ~supervisely.api.my_module_api.MyModuleApi
       ~supervisely.api.my_module_api.MyModuleInfo
   ```

Existing section order in `sdk_packages.rst`:
`API` → `Video API` → `Volume API` → `Pointcloud API` → `Entity API` → `Neural Networks API` → …

### When updating an existing module

- If a new public class or NamedTuple is added — add it to both places above.
- If a class is renamed or removed — update both places accordingly.
- If a module moves to a sub-package — update the dotted path in `sdk_packages.rst`.

---

## Checklist

- [ ] File starts with `# coding: utf-8` + module docstring + `from __future__ import annotations`
- [ ] `*Info` NamedTuple defined with required fields first, optional with defaults, guard comment at end
- [ ] `info_sequence()` and `info_tuple_name()` implemented
- [ ] `_convert_json_info` only overridden when normalization is needed; calls `super()` if so
- [ ] Module registered in `Api.__init__` in `api.py`
- [ ] Public symbols exported in `supervisely/__init__.py`
- [ ] **`docs/source/sdk_packages.rst` updated** — both in "API" section and in a dedicated section
