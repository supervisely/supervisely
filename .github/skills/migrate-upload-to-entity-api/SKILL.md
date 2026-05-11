---
name: migrate-upload-to-entity-api
description: >
  Migrate upload_hashes / upload_links / upload_ids / _upload_bulk_add from type-specific
  media modules (video_api, pointcloud_api, pointcloud_episode_api, volume_api) to delegate
  to api.entity.*. Use when asked to integrate entity_api into old modules, replace _upload_bulk_add
  with entity calls, or migrate bulk-add logic to the generic entity endpoint.
---

# Migrate upload methods to `api.entity`

## Context

`supervisely/api/entity_api.py` is a fully-implemented generic upload/list/info layer
that calls `entities.bulk.add`, `entities.list`, `entities.info`, and `entities.download`.

The old modules (`video_api.py`, `pointcloud_api.py`, `pointcloud_episode_api.py`,
`volume_api.py`) each have their own `_upload_bulk_add` / `upload_hashes` / `upload_links`
that call the legacy type-specific endpoints (`videos.bulk.add`, etc.).

The goal is to keep the public API signatures **unchanged** while delegating the actual
HTTP call to `self._api.entity.*`.

---

## Key types

| Symbol | Location |
|--------|----------|
| `EntityInfo` | `supervisely/api/entity_api.py` — NamedTuple (23 fields, no `title`) |
| `EntityDescriptor` | `supervisely/api/entity_api.py` — builder with 5 class-methods |
| `EntityApi` | `supervisely/api/entity_api.py` — `self._api.entity` in `api.py` |

### `EntityApi` public methods

| Method | Server endpoint | Notes |
|--------|----------------|-------|
| `add(dataset_id, entities, generate_unique_names, force_metadata_for_links, skip_validation, progress_cb)` | `entities.bulk.add` | Accepts `List[EntityDescriptor \| dict]` |
| `get_info_by_id(id, fields, omit_frames_to_timecodes, force_metadata_for_links)` | `entities.info` | Returns `EntityInfo` |
| `get_list(dataset_id, project_id, ...)` | `entities.list` | Full pagination |
| `get_list_generator(dataset_id, project_id, ...)` | `entities.list` | Token-based pagination |
| `download(id)` | `entities.download` | Returns raw `bytes` |

There are **no** `add_hashes`, `add_links`, `add_entity_ids` etc. — use `EntityDescriptor`
class-methods + `add()` directly.

### `EntityDescriptor` class-methods

| Method | Sends field |
|--------|-------------|
| `EntityDescriptor.from_hash(hash, name, **kw)` | `hash` |
| `EntityDescriptor.from_link(link, name, **kw)` | `link` |
| `EntityDescriptor.from_entity_id(entity_id, name, **kw)` | `entityId` |
| `EntityDescriptor.from_team_file(team_file_id, name, **kw)` | `teamFileId` |
| `EntityDescriptor.from_source_blob(blob_id, offset_start, offset_end, name, **kw)` | `entityId` + `sourceBlob` |

`**kw` optional fields: `description`, `meta`, `parent_id`, `created_at`, `updated_at`, `created_by`.

### Important: field naming

- The server returns `name` (not `title`). `EntityApi` always uses `name`.
- `_normalize_entity_payload` only normalises `created_by` → `createdBy` and
  snake_case keys inside `sourceBlob`. It does **not** touch `name`/`title`.

---

## Migration pattern for `_upload_bulk_add`

### Before (video_api.py pattern)
```python
def _upload_bulk_add(self, func_item_to_kv, dataset_id, names, items, metas=None, progress_cb=None, force_metadata_for_links=True):
    if metas is None:
        metas = [{}] * len(items)
    results = []
    for batch in batched(list(zip(names, items, metas))):
        videos = []
        for name, item, meta in batch:
            item_tuple = func_item_to_kv(item)
            videos.append({"title": name, item_tuple[0]: item_tuple[1], ApiField.META: meta or {}})
        response = self._api.post(
            "videos.bulk.add",
            {ApiField.DATASET_ID: dataset_id, ApiField.VIDEOS: videos, ...},
        )
        results.extend(self._convert_json_info(item) for item in response.json())
    name_to_res = {img_info.name: img_info for img_info in results}
    return [name_to_res[name] for name in names]
```

### After — Option A: raw JSON path (recommended, zero regression risk)

Expose a thin helper in `EntityApi` that returns raw JSON instead of `EntityInfo`:

```python
# Add to EntityApi in entity_api.py:
def _raw_add(self, dataset_id, entities, *, force_metadata_for_links=True,
             generate_unique_names=False, skip_validation=False) -> list:
    """Like add(), but returns the raw server JSON list."""
    data = {
        ApiField.DATASET_ID: dataset_id,
        ApiField.ENTITIES: [self._normalize_entity_payload(e) for e in entities],
        ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
        ApiField.SKIP_VALIDATION: skip_validation,
    }
    if generate_unique_names:
        data[ApiField.GENERATE_UNIQUE_NAMES] = generate_unique_names
    return self._api.post("entities.bulk.add", data).json()
```

Then in each module's `_upload_bulk_add`:

```python
def _upload_bulk_add(self, func_item_to_kv, dataset_id, names, items, metas=None, progress_cb=None, force_metadata_for_links=True):
    if metas is None:
        metas = [{}] * len(items)
    results = []
    for batch in batched(list(zip(names, items, metas))):
        entities = []
        for name, item, meta in batch:
            field, value = func_item_to_kv(item)
            payload = {"name": name, field: value}
            if meta:
                payload[ApiField.META] = meta
            entities.append(payload)
        raw = self._api.entity._raw_add(
            dataset_id, entities,
            force_metadata_for_links=force_metadata_for_links,
        )
        if progress_cb is not None:
            progress_cb(len(raw))
        results.extend(self._convert_json_info(item) for item in raw)
    name_to_res = {info.name: info for info in results}
    return [name_to_res[name] for name in names]
```

This keeps each module's own `_convert_json_info` intact so video-specific fields
(`frames_count`, `frame_width`, `tags`, etc.) continue to work.

### After — Option B: `EntityDescriptor` + `add()` (for new callers only)

Only viable when the caller does **not** need the module's typed `*Info` back:

```python
entities = [
    EntityDescriptor.from_hash(h, name=n, meta=m)
    for n, h, m in zip(names, hashes, metas or [{}] * len(names))
]
return self._api.entity.add(
    dataset_id=dataset_id,
    entities=entities,
    force_metadata_for_links=force_metadata_for_links,
    progress_cb=progress_cb,
)
# Returns List[EntityInfo], NOT List[VideoInfo] etc.
```

---

## Module-by-module instructions

### `video_api.py`

1. Replace the `self._api.post("videos.bulk.add", ...)` call inside `_upload_bulk_add`
   with `self._api.entity._raw_add(...)` (Option A above).
2. Change `"title": name` → `"name": name` in the payload dict.
3. Keep the `None`-hash validation guard in `upload_hashes`.
4. `upload_ids` calls `upload_hashes` / `upload_links` internally — no further changes.

---

### `pointcloud_api.py`

Same as `video_api.py`. The `meta` dict (e.g. `{"frame": N}`) is passed through unchanged.

---

### `pointcloud_episode_api.py`

Point cloud episodes **require** `meta={"frame": N}` for every entity.
Add a guard before calling `_raw_add`:

```python
for i, meta in enumerate(metas or []):
    if not isinstance(meta, dict) or "frame" not in meta:
        raise ValueError(f"metas[{i}] must contain 'frame' key for episode entities")
```

---

### `volume_api.py`

Same pattern as `video_api.py`. No frame/timecode fields, so the raw JSON from
`entities.bulk.add` maps cleanly to `VolumeInfo`.

---

## Batching

Old modules use `sly.batched(...)` inside `_upload_bulk_add`. Keep that batching —
`_raw_add` does **not** batch internally (unlike `add()`).

---

## Checklist for each module

- [ ] Read the current `_upload_bulk_add` implementation
- [ ] Add `_raw_add` to `EntityApi` if not already present
- [ ] Replace `self._api.post("<type>.bulk.add", ...)` with `self._api.entity._raw_add(...)`
- [ ] Change `"title": name` → `"name": name` in the entity payload dict
- [ ] Keep existing input validation (null-hash guard, extension check, etc.)
- [ ] Keep `_convert_json_info` and the `name_to_res` ordering logic unchanged
- [ ] Run existing tests for that module
