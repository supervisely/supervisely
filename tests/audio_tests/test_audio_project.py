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
                settings=sly.SpectrogramSettings(scale="mel", channel=0),
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
    # The labeled channel and the viewed channel stay separate on disk too.
    assert segment.channel == 1
    assert segment.settings.channel == 0
    assert segment.settings.scale == "mel"


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
