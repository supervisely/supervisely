# coding: utf-8
"""Local audio project: on-disk layout, annotation round trip, dispatch."""

import math
import struct
import wave

import pytest

import supervisely as sly
from supervisely.project import get_project_class
from supervisely.project.download import _add_save_items_infos_to_kwargs
from supervisely.project.project_type import ProjectType

SR = 16000


def write_wav(path, seconds=1.0, channels=1, sample_rate=SR):
    n = int(sample_rate * seconds)
    frames = bytearray()
    for i in range(n):
        for ch in range(channels):
            frames += struct.pack(
                "<h", int(18000 * math.sin(2 * math.pi * 440 * (ch + 1) * i / sample_rate))
            )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))
    return str(path)


@pytest.fixture
def meta():
    return sly.ProjectMeta(
        tag_metas=sly.TagMetaCollection([sly.TagMeta("Event", sly.TagValueType.NONE)])
    )


def test_audio_project_type_is_dispatched():
    assert get_project_class("audio") is sly.AudioProject
    kwargs = _add_save_items_infos_to_kwargs({}, str(ProjectType.AUDIO))
    assert kwargs["save_audio_info"] is True


def test_project_layout_and_round_trip(tmp_path, meta):
    audio_path = write_wav(tmp_path / "rain.wav", seconds=2.0, channels=2)

    project = sly.AudioProject(str(tmp_path / "proj"), sly.OpenMode.CREATE)
    project.set_meta(meta)
    dataset = project.create_dataset("ds0")

    ann = sly.AudioAnnotation(
        sample_count=2 * SR,
        sample_rate=SR,
        channels=2,
        tags=[
            sly.AudioSegment(
                name="Event",
                start=0,
                end=15999,
                channel=1,
                meta={"reviewedBy": "anna"},
            )
        ],
    )
    dataset.add_item_file("rain.wav", audio_path, ann=ann)

    reopened = sly.AudioProject(str(tmp_path / "proj"), sly.OpenMode.READ)
    ds = reopened.datasets.get("ds0")
    assert ds.get_items_names() == ["rain.wav"]
    assert ds.item_dir.endswith("ds0/audio")
    assert ds.get_ann_path("rain.wav").endswith("ds0/ann/rain.wav.json")

    restored = ds.get_ann("rain.wav", reopened.meta)
    assert (restored.sample_count, restored.sample_rate, restored.channels) == (2 * SR, SR, 2)
    assert restored.duration_seconds == pytest.approx(2.0)

    segment = restored.tags[0]
    assert segment.name == "Event"
    assert (segment.start, segment.end) == (0, 15999)
    assert segment.channel == 1
    assert segment.meta == {"reviewedBy": "anna"}


def test_empty_annotation_reads_the_shape_from_the_file(tmp_path, meta):
    audio_path = write_wav(tmp_path / "beep.wav", seconds=0.5, channels=1)

    project = sly.AudioProject(str(tmp_path / "proj"), sly.OpenMode.CREATE)
    project.set_meta(meta)
    dataset = project.create_dataset("ds0")
    dataset.add_item_file("beep.wav", audio_path)

    ann = dataset.get_ann("beep.wav", project.meta)
    assert ann.tags == []
    assert ann.sample_count == SR // 2
    assert ann.sample_rate == SR
    assert ann.channels == 1


def test_unknown_tag_is_rejected(tmp_path, meta):
    ann_json = sly.AudioAnnotation(
        tags=[sly.AudioSegment(name="Nope", start=0, end=1)]
    ).to_json()
    with pytest.raises(RuntimeError):
        sly.AudioAnnotation.from_json(ann_json, meta)


def test_non_audio_file_is_refused(tmp_path, meta):
    junk = tmp_path / "notes.txt"
    junk.write_text("hello")

    project = sly.AudioProject(str(tmp_path / "proj"), sly.OpenMode.CREATE)
    project.set_meta(meta)
    dataset = project.create_dataset("ds0")
    with pytest.raises(Exception):
        dataset.add_item_file("notes.txt", str(junk))


def test_meta_json_carries_the_project_spectrogram(tmp_path, meta):
    """The analysis the labels were drawn under is project configuration, so it
    travels in meta.json -- otherwise a downloaded project cannot be rendered
    the way the annotators saw it."""
    settings = sly.SpectrogramSettings(scale="mel", fft_size=1024, mel_bands=64)
    meta_with_spectrogram = meta.clone(
        project_settings=sly.ProjectSettings(spectrogram=settings.to_json())
    )

    audio_path = write_wav(tmp_path / "rain.wav", seconds=0.5)
    project = sly.AudioProject(str(tmp_path / "proj"), sly.OpenMode.CREATE)
    project.set_meta(meta_with_spectrogram)
    project.create_dataset("ds0").add_item_file("rain.wav", audio_path)

    reopened = sly.AudioProject(str(tmp_path / "proj"), sly.OpenMode.READ)
    restored = sly.SpectrogramSettings.from_json(reopened.meta.project_settings.spectrogram)
    assert restored == settings
    assert restored.fingerprint == settings.fingerprint


def test_project_settings_without_a_spectrogram_stay_clean(meta):
    """Every other modality shares this class; an unset spectrogram must not
    appear in their meta.json."""
    assert "spectrogram" not in sly.ProjectSettings().to_json()
    assert meta.project_settings.spectrogram is None


def test_recording_tags_are_stored_next_to_segments(tmp_path, meta):
    """On disk a whole-recording tag is an entry of `tags` with
    `frameRange: null` -- the platform's own shape."""
    meta = meta.add_tag_meta(sly.TagMeta("Scene", sly.TagValueType.ANY_STRING))
    audio_path = write_wav(tmp_path / "rain.wav", seconds=0.5)
    project = sly.AudioProject(str(tmp_path / "proj"), sly.OpenMode.CREATE)
    project.set_meta(meta)
    dataset = project.create_dataset("ds0")
    ann = sly.AudioAnnotation(
        tags=[sly.AudioSegment(name="Event", start=0, end=99)],
        recording_tags=[
            sly.AudioRecordingTag(name="Scene", value="indoor", custom_data={"k": 1})
        ],
    )
    dataset.add_item_file("rain.wav", audio_path, ann=ann)

    on_disk = sly.io.json.load_json_file(dataset.get_ann_path("rain.wav"))
    assert {"name": "Scene", "frameRange": None, "value": "indoor", "customData": {"k": 1}} in on_disk["tags"]

    restored = sly.AudioProject(str(tmp_path / "proj"), sly.OpenMode.READ).datasets.get("ds0")
    restored = restored.get_ann("rain.wav", project.meta)
    assert [(t.name, t.value, t.custom_data) for t in restored.recording_tags] == [
        ("Scene", "indoor", {"k": 1})
    ]
    assert [(s.start, s.end) for s in restored.tags] == [(0, 99)]


def test_unknown_recording_tag_is_rejected(meta):
    ann_json = sly.AudioAnnotation(recording_tags=[sly.AudioRecordingTag(name="Nope")]).to_json()
    with pytest.raises(RuntimeError):
        sly.AudioAnnotation.from_json(ann_json, meta)


def test_download_keeps_recording_tags(tmp_path):
    from supervisely.api.audio_api import AudioApi
    from supervisely.project.audio_project import _build_annotation

    info = AudioApi(None)._convert_json_info(
        {
            "id": 5,
            "name": "rain.wav",
            "tags": [
                {"id": 1, "tagId": 10, "value": "calm", "meta": {"channel": None}},
                {"id": 2, "tagId": 11, "frameRange": [0, 99], "meta": {"channel": 1}},
            ],
        }
    )
    ann = _build_annotation(info, str(tmp_path / "x.wav"), {10: "Mood", 11: "Event"}, False)
    assert [(t.name, t.value) for t in ann.recording_tags] == [("Mood", "calm")]
    assert [(s.name, s.start, s.channel) for s in ann.tags] == [("Event", 0, 1)]


class _UploadApi:
    """Just enough of the API for upload_audio_project."""

    def __init__(self):
        from types import SimpleNamespace

        self.ns = SimpleNamespace
        self.datasets = []
        self.tags = {}
        self.meta = None
        api = self

        class _Project:
            def create(self, workspace_id, name, type=None, change_name_if_conflict=False):
                return api.ns(id=1, name=name)

            def update_meta(self, id, meta):
                api.meta = {**meta, "tags": [{**t, "id": 100 + i} for i, t in enumerate(meta["tags"])]}

            def get_meta(self, id, with_settings=False):
                return api.meta

        class _Dataset:
            def create(self, project_id, name, change_name_if_conflict=False, parent_id=None):
                if "/" in name:
                    raise ValueError(f"the server refuses {name!r}")
                api.datasets.append((len(api.datasets) + 10, name, parent_id))
                return api.ns(id=api.datasets[-1][0], name=name)

        class _Audio:
            def upload_paths(self, dataset_id, names, paths, progress_cb=None):
                return [api.ns(id=dataset_id * 100 + i, name=n) for i, n in enumerate(names)]

            def add_tags(self, project_id, entity_id, segments=None, recording_tags=None):
                api.tags[entity_id] = (
                    [s.tag_id for s in segments or []],
                    [t.tag_id for t in recording_tags or []],
                )

        self.project, self.dataset, self.audio = _Project(), _Dataset(), _Audio()


def test_upload_recreates_nested_datasets_and_recording_tags(tmp_path, meta):
    """`sly.upload` of a project with `recordings/night` used to send the
    path-style name to the server, which answers 400."""
    meta = meta.add_tag_meta(sly.TagMeta("Scene", sly.TagValueType.ANY_STRING))
    project = sly.AudioProject(str(tmp_path / "proj"), sly.OpenMode.CREATE)
    project.set_meta(meta)
    parent = project.create_dataset("recordings")
    child = project.create_dataset("night", "recordings/datasets/night")
    for i, ds in enumerate((parent, child)):
        ds.add_item_file(
            "a.wav",
            write_wav(tmp_path / f"a{i}.wav", seconds=0.1),
            ann=sly.AudioAnnotation(
                tags=[sly.AudioSegment(name="Event", start=0, end=9)],
                recording_tags=[sly.AudioRecordingTag(name="Scene", value="x")],
            ),
        )

    api = _UploadApi()
    sly.upload_audio_project(str(tmp_path / "proj"), api, 1, log_progress=False)

    (parent_id, parent_name, grandparent), (_, child_name, child_parent) = api.datasets
    assert (parent_name, grandparent) == ("recordings", None)
    assert (child_name, child_parent) == ("night", parent_id)
    assert list(api.tags.values()) == [([100], [101])] * 2
