"""Tests for image audio references and the project-dir resolution they depend on.

Before the fix, `find_project_dirs` never yielded a project directory passed directly to it
(`os.path.join(dir, "")` left a trailing separator and `Project()` raised), so `upload_project`
was always handed an ancestor of the project - and then looked for the per-item `meta` dir
under that ancestor, silently dropping every image meta including audio references.
"""

import os

import numpy as np
import pytest

import supervisely as sly
from supervisely.api.image_api import AudioReference, ImageApi
from supervisely.api.module_api import ApiField
from supervisely.project.project import find_project_dirs

CANONICAL = [
    {"url": "https://instance/a.mp3", "name": "Operator note", "mimeType": "audio/mpeg"},
    {"url": "https://instance/b.mp3", "name": "Second pass"},
]


def test_to_json_omits_unset_optional_fields():
    assert AudioReference(url="https://instance/a.mp3").to_json() == {
        "url": "https://instance/a.mp3"
    }
    assert AudioReference(url="https://instance/a.mp3", name="A").to_json() == {
        "url": "https://instance/a.mp3",
        "name": "A",
    }


def test_from_json_to_json_round_trip_is_identity():
    assert [AudioReference.from_json(e).to_json() for e in CANONICAL] == CANONICAL


@pytest.mark.parametrize(
    "data",
    [
        {"name": "no url"},
        {"url": ""},
        {"url": None},
        {"audios": [{"link": "https://instance/a.mp3"}]},
    ],
)
def test_from_json_requires_a_non_empty_url(data):
    with pytest.raises(ValueError):
        AudioReference.from_json(data)


@pytest.mark.parametrize("data", ["https://instance/a.mp3", ["https://instance/a.mp3"], None])
def test_from_json_rejects_non_dict(data):
    with pytest.raises(TypeError):
        AudioReference.from_json(data)


def test_parse_reads_the_canonical_form():
    refs = ImageApi._parse_audio_references({ApiField.AUDIO: CANONICAL})
    assert [(r.name, r.mime_type) for r in refs] == [
        ("Operator note", "audio/mpeg"),
        ("Second pass", None),
    ]


@pytest.mark.parametrize("meta", [{}, None, {ApiField.AUDIO: None}])
def test_parse_returns_empty_without_audio(meta):
    assert ImageApi._parse_audio_references(meta) == []


def test_parse_ignores_a_non_list_value():
    assert ImageApi._parse_audio_references({ApiField.AUDIO: "https://instance/a.mp3"}) == []


def test_parse_skips_malformed_entries_instead_of_raising():
    meta = {ApiField.AUDIO: [{"name": "broken, no url"}, CANONICAL[0]]}
    refs = ImageApi._parse_audio_references(meta)
    assert [r.url for r in refs] == [CANONICAL[0]["url"]]


def test_update_audio_references_is_pure_and_keeps_other_keys():
    source = {"Camera Make": "Canon"}
    updated = ImageApi.update_audio_references(source, AudioReference(url="https://instance/a.mp3"))

    assert source == {"Camera Make": "Canon"}, "source meta must not be mutated"
    assert updated["Camera Make"] == "Canon"
    assert updated[ApiField.AUDIO] == [{"url": "https://instance/a.mp3"}]


@pytest.mark.parametrize(
    "references",
    [
        AudioReference(url="https://instance/a.mp3", name="A"),
        {"url": "https://instance/a.mp3", "name": "A"},
        [AudioReference(url="https://instance/a.mp3", name="A")],
        [{"url": "https://instance/a.mp3", "name": "A"}],
    ],
)
def test_update_audio_references_accepts_objects_dicts_and_single_values(references):
    updated = ImageApi.update_audio_references({}, references)
    assert updated[ApiField.AUDIO] == [{"url": "https://instance/a.mp3", "name": "A"}]


def test_update_audio_references_with_empty_list_clears_them():
    updated = ImageApi.update_audio_references({ApiField.AUDIO: CANONICAL}, [])
    assert updated[ApiField.AUDIO] == []


def test_update_audio_references_validates_before_writing():
    with pytest.raises(ValueError):
        ImageApi.update_audio_references({}, {"name": "no url"})


def _make_project(path: str) -> str:
    """Create a minimal readable project on disk. Project(READ) rejects an empty one."""
    project = sly.Project(path, sly.OpenMode.CREATE)
    dataset = project.create_dataset("ds")
    dataset.add_item_np("img.png", np.zeros((2, 2, 3), dtype=np.uint8))
    return path


def test_find_project_dirs_accepts_the_project_dir_itself(tmp_path):
    project_dir = _make_project(str(tmp_path / "proj"))
    assert list(find_project_dirs(project_dir)) == [project_dir]


def test_find_project_dirs_tolerates_a_trailing_separator(tmp_path):
    project_dir = _make_project(str(tmp_path / "proj"))
    assert list(find_project_dirs(project_dir + os.sep)) == [project_dir]


def test_find_project_dirs_still_finds_a_nested_project(tmp_path):
    project_dir = _make_project(str(tmp_path / "parent" / "proj"))
    assert list(find_project_dirs(str(tmp_path))) == [project_dir]
