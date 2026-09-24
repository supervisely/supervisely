# coding: utf-8
"""Auto Import for audio: format detection and upload through ImportManager's converter."""

import math
import struct
import wave
from types import SimpleNamespace

import pytest

import supervisely as sly
from supervisely.convert.audio.audio_converter import AudioConverter
from supervisely.convert.audio.sly.sly_audio_converter import SLYAudioConverter
from supervisely.convert.converter import ImportManager
from supervisely.project.project_settings import LabelingInterface
from supervisely.project.project_type import ProjectType

SR = 16000


def write_wav(path, seconds=0.25, channels=1):
    n = int(SR * seconds)
    frames = bytearray()
    for i in range(n):
        for _ in range(channels):
            frames += struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / SR)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(frames))
    return str(path)


def make_project(root, datasets=("ds0",), spectrogram=None, tags=("Event",)):
    meta = sly.ProjectMeta(
        tag_metas=sly.TagMetaCollection([sly.TagMeta(t, sly.TagValueType.NONE) for t in tags]),
        project_type=ProjectType.AUDIO.value,
        project_settings=sly.ProjectSettings(spectrogram=spectrogram),
    )
    project = sly.AudioProject(str(root), sly.OpenMode.CREATE)
    project.set_meta(meta)
    for ds_name in datasets:
        dataset = project.create_dataset(ds_name)
        wav = write_wav(root.parent / f"{ds_name.replace('/', '_')}.wav")
        ann = sly.AudioAnnotation(
            sample_count=SR // 4,
            sample_rate=SR,
            channels=1,
            tags=[sly.AudioSegment(name=tags[0], start=10, end=2000, channel=0, tag_id=999, id=7)],
        )
        dataset.add_item_file("rec.wav", wav, ann=ann)
    return str(root)


def detect(input_dir, upload_as_links=False):
    return AudioConverter(input_dir, upload_as_links=upload_as_links).detect_format()


class FakeApi:
    """Just enough of the API for AudioConverter.upload_dataset."""

    def __init__(self, spectrogram=None, items_count=0, tags=()):
        self.settings = {} if spectrogram is None else {"spectrogram": spectrogram}
        self.meta = {"classes": [], "tags": [{"id": 100 + i, "name": t, "value_type": "none"} for i, t in enumerate(tags)]}
        self.items_count = items_count
        self.uploaded = []
        self.segments = {}
        self.spectrogram_set = None
        self.datasets = {1: "ds"}
        api = self

        class _Project:
            def get_meta(self, id, with_settings=False):
                return {**api.meta, "projectType": "audio"}

            def update_meta(self, id, meta):
                meta_json = meta.to_json() if not isinstance(meta, dict) else meta
                known = {t["name"]: t["id"] for t in api.meta["tags"]}
                next_id = 100 + len(known)
                tags = []
                for tag in meta_json["tags"]:
                    if tag["name"] not in known:
                        known[tag["name"]] = next_id
                        next_id += 1
                    tags.append({**tag, "id": known[tag["name"]]})
                api.meta = {**meta_json, "tags": tags}
                return sly.ProjectMeta.from_json(api.meta)

            def get_settings(self, id):
                return dict(api.settings)

            def get_info_by_id(self, id):
                return SimpleNamespace(id=id, items_count=api.items_count)

        class _Dataset:
            def get_info_by_id(self, id, raise_error=False):
                return SimpleNamespace(id=id, project_id=5, name=api.datasets[id])

            def get_list(self, project_id, recursive=False):
                return [SimpleNamespace(id=k, name=v) for k, v in api.datasets.items()]

            def update(self, id, name):
                api.datasets[id] = name

            def create(self, project_id, name, parent_id=None):
                new_id = max(api.datasets) + 1
                api.datasets[new_id] = name
                return SimpleNamespace(id=new_id, name=name)

        class _Audio:
            def get_list(self, dataset_id, recursive=True):
                return [SimpleNamespace(name=n) for d, n, _ in api.uploaded if d == dataset_id]

            def upload_paths(self, dataset_id, names, paths, progress_cb=None):
                infos = []
                for name, path in zip(names, paths):
                    api.uploaded.append((dataset_id, name, path))
                    infos.append(SimpleNamespace(id=len(api.uploaded), name=name))
                return infos

            def add_segments(self, project_id, entity_id, segments):
                api.segments[entity_id] = [s._to_api_json(entity_id) for s in segments]

            def set_spectrogram_settings(self, project_id, settings):
                api.spectrogram_set = settings
                api.settings["spectrogram"] = settings.to_json()

        self.project = _Project()
        self.dataset = _Dataset()
        self.audio = _Audio()


def test_import_manager_dispatches_audio(tmp_path):
    write_wav(tmp_path / "a.wav")
    manager = ImportManager.__new__(ImportManager)
    manager._modality = ProjectType.AUDIO.value
    manager._input_data = str(tmp_path)
    manager._labeling_interface = LabelingInterface.DEFAULT
    manager._upload_as_links = False
    manager._remote_files_map = {}
    manager._team_files_id_map = {}
    assert isinstance(manager.get_converter(), AudioConverter)


def test_raw_recordings_in_any_structure(tmp_path):
    (tmp_path / "nested" / "deeper").mkdir(parents=True)
    write_wav(tmp_path / "a.wav")
    write_wav(tmp_path / "nested" / "deeper" / "b.wav")
    (tmp_path / "notes.txt").write_text("not audio")

    converter = detect(str(tmp_path))
    assert type(converter) is AudioConverter
    assert sorted(item.name for item in converter.get_items()) == ["a.wav", "b.wav"]


def test_no_recordings_is_an_error(tmp_path):
    (tmp_path / "notes.txt").write_text("not audio")
    with pytest.raises(RuntimeError):
        detect(str(tmp_path))


def test_supervisely_project_is_detected(tmp_path):
    settings = sly.SpectrogramSettings(scale="mel", fft_size=1024, mel_bands=64)
    make_project(tmp_path / "proj", spectrogram=settings.to_json())

    converter = detect(str(tmp_path / "proj"))
    assert isinstance(converter, SLYAudioConverter)
    assert converter.items_count == 1
    assert converter.get_meta().project_settings.spectrogram == settings.to_json()


def test_links_request_does_not_hide_the_format(tmp_path, monkeypatch):
    """Audio is always transferred; asking for links must not make detection skip
    the Supervisely format and import the recordings without their labels."""
    monkeypatch.setattr("supervisely.convert.base_converter.Api.from_env", lambda: None)
    monkeypatch.setattr("supervisely.convert.base_converter.team_id", lambda: 1)
    make_project(tmp_path / "proj")
    assert isinstance(detect(str(tmp_path / "proj"), upload_as_links=True), SLYAudioConverter)


def test_other_modality_project_is_not_taken_for_audio(tmp_path):
    project = sly.Project(str(tmp_path / "img"), sly.OpenMode.CREATE)
    project.set_meta(sly.ProjectMeta())
    project.create_dataset("ds0")
    write_wav(tmp_path / "img" / "stray.wav")
    assert type(detect(str(tmp_path / "img"))) is AudioConverter


def test_upload_writes_segments_against_destination_tag_ids(tmp_path):
    make_project(tmp_path / "proj")
    converter = detect(str(tmp_path / "proj"))
    api = FakeApi()

    converter.upload_dataset(api, 1, log_progress=False)

    assert [(d, n) for d, n, _ in api.uploaded] == [(1, "rec.wav")]
    (payload,) = api.segments[1]
    assert payload["tagId"] == 100  # the destination's id, not the exported 999
    assert payload["frameRange"] == [10, 2000]
    assert payload["meta"] == {"channel": 0}


def test_conflicting_tag_is_renamed_and_followed(tmp_path):
    make_project(tmp_path / "proj")
    converter = detect(str(tmp_path / "proj"))
    api = FakeApi(tags=())
    api.meta["tags"] = [{"id": 50, "name": "Event", "value_type": "any_string"}]

    converter.upload_dataset(api, 1, log_progress=False)

    renamed_id = {t["name"]: t["id"] for t in api.meta["tags"]}["Event_1"]
    assert api.segments[1][0]["tagId"] == renamed_id


def test_segment_with_unknown_tag_is_skipped(tmp_path):
    make_project(tmp_path / "proj")
    converter = detect(str(tmp_path / "proj"))
    item = converter.get_items()[0]
    ann = converter.to_supervisely(item)
    ann.tags[0].name = "Missing"
    assert AudioConverter._resolve_tag_ids("rec.wav", ann.tags, {"Event": 1}) == []


def test_name_conflict_in_dataset_gets_a_free_name(tmp_path):
    make_project(tmp_path / "proj")
    converter = detect(str(tmp_path / "proj"))
    api = FakeApi()
    api.uploaded.append((1, "rec.wav", "existing"))

    converter.upload_dataset(api, 1, log_progress=False)
    assert api.uploaded[-1][1] == "rec_01.wav"


def test_spectrogram_applied_to_an_unconfigured_empty_project(tmp_path):
    settings = sly.SpectrogramSettings(scale="mel", fft_size=1024, mel_bands=64)
    make_project(tmp_path / "proj", spectrogram=settings.to_json())
    api = FakeApi()

    detect(str(tmp_path / "proj")).upload_dataset(api, 1, log_progress=False)
    assert api.spectrogram_set == settings


def test_spectrogram_never_changed_under_existing_labels(tmp_path):
    imported = sly.SpectrogramSettings(scale="mel", fft_size=1024, mel_bands=64)
    make_project(tmp_path / "proj", spectrogram=imported.to_json())

    configured = FakeApi(spectrogram=sly.SpectrogramSettings(scale="log").to_json())
    detect(str(tmp_path / "proj")).upload_dataset(configured, 1, log_progress=False)
    assert configured.spectrogram_set is None

    unconfigured_with_items = FakeApi(items_count=3)
    detect(str(tmp_path / "proj")).upload_dataset(unconfigured_with_items, 1, log_progress=False)
    assert unconfigured_with_items.spectrogram_set is None


def test_raw_upload_leaves_spectrogram_alone(tmp_path):
    write_wav(tmp_path / "a.wav")
    api = FakeApi()
    detect(str(tmp_path)).upload_dataset(api, 1, log_progress=False)
    assert api.spectrogram_set is None
    assert api.segments == {}


def test_nested_datasets_keep_their_hierarchy(tmp_path):
    make_project(tmp_path / "proj", datasets=("first", "second"))
    converter = detect(str(tmp_path / "proj"))
    api = FakeApi()

    converter.upload_dataset(api, 1, log_progress=False)
    assert sorted(api.datasets.values()) == ["first", "second"]
    assert len(api.uploaded) == 2
    assert all(len(segments) == 1 for segments in api.segments.values())
