Title: SDK: minor bugfixes

Description:
- Fix VideoApi file-size error handling for the newer `100mb` payload format.
- Preserve `dataset_id` in empty point-cloud episode annotations.
- Hydrate storage-backed point-cloud figure indices after figure download.
- Resolve tag names from `tagId` when annotation payloads omit `name`.
- Validate video tag frame ranges against video bounds.
- Parse `maxModules` in team `UsageInfo`.
