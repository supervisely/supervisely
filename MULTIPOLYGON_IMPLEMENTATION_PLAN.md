# Multipolygon End-to-End Implementation Plan

## SDK

- [x] Review existing geometry patterns and keep Multipolygon consistent with Polygon and other geometry classes.
- [x] Add `Multipolygon` geometry class with JSON serialization/deserialization, drawing, area, bbox, crop, transforms, approximation, and conversion to several `Polygon` objects.
- [x] Register `Multipolygon` in geometry exports and JSON shape dispatch so project meta and annotations deserialize correctly.
- [x] Update SDK widgets/helpers that list supported class shape names where needed.
- [x] Add local unit/verification tests for serialization, annotation/meta round trips, drawing/masks, transforms, crops, and conversion to polygons.
- [x] Add an ignored real API verification script for dev instance operations: receive existing multipolygons, create/update meta, upload/download annotations, and render output images.
- [x] Run local tests and real dev-instance verification. Local tests pass. Receiving/rendering existing dev multipolygons, creating project/meta, uploading/downloading a two-part multipolygon annotation, and rendering downloaded output pass. `annotations.bulk.add` returns API 500 for inline multipolygon payloads on dev, so SDK upload routes multipolygon labels through `figures.bulk.add`.

## Auto Import / Main Import

- [x] Review Supervisely-format import flow in `C:\Supervisely\multipolygon\main-import` and upstream repository.
- [x] Confirm it relies on SDK geometry parsing or update shape allowlists/mappings for `multipolygon`.
- [x] Run a local verification path with Supervisely-format data that contains Multipolygon. `SLYImageConverter` detects the sample, creates multipolygon meta, and converts annotation labels to `sly.Multipolygon`; full `ImportManager` detection needs the app dependency set from `dev_requirements.txt`.

## Docs

- [x] Update `C:\Supervisely\multipolygon\docs` with Multipolygon geometry information.
- [x] Update `C:\Supervisely\multipolygon\developer-portal` with Multipolygon geometry information.
- [x] Add a Developer Portal Python SDK tutorial showing how to create, serialize, draw, upload/download, and convert Multipolygon objects.

## Export to Supervisely Format

- [x] Review `C:\Supervisely\multipolygon\export-to-supervisely-format`.
- [x] Update SDK dependency/Docker image references for the next SDK version, assumed `6.74.10` unless repo state says otherwise.
- [x] Verify export preserves Multipolygon meta and annotations.

## PR Preparation

- [ ] Create clean branches and commits from the user GitHub account only, with no AI/coauthor metadata.
- [ ] Prepare PR for SDK.
- [ ] Prepare PR for Docs.
- [ ] Prepare PR for Developer Portal.
- [ ] Prepare PR for Main Import / Auto Import.
- [ ] Prepare PR for Export to Supervisely Format.
- [ ] Write final report with completed work, verification results, known issues, and manual follow-up steps.
