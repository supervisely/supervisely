# Audio modality

Audio projects hold recordings labeled with **time segments**. Each segment can
record the spectrogram settings it was drawn under, so the picture the annotator
was looking at can be reproduced later and fed to training.

```python
import supervisely as sly

api = sly.Api.from_env()

project = api.project.create(workspace_id, "engine-noise", type=sly.ProjectType.AUDIO)
dataset = api.dataset.create(project.id, "bench-run-1")

recording = api.audio.upload_path(dataset.id, "run1.wav", "/data/run1.wav")
recording.id, recording.name, recording.file_meta
```

`api.audio` returns `AudioInfo` named tuples, like every other modality. Uploads
are de-duplicated by hash, so re-running an interrupted `upload_paths` sends only
what is missing, and take a `progress_cb`.

## Segments are inclusive sample ranges

A segment is a tag applied to a range of the recording. The two numbers are
**inclusive, zero-based indices into the original source samples** — not frames
and not milliseconds, despite the `frameRange` name inherited from video. They
are 64-bit, so long recordings at high sample rates do not overflow.

```python
segment = sly.AudioSegment(tag_id=tag_id, start=16000, end=31999, channel=0)
segment.sample_count            # 16000 -- both endpoints counted
segment.duration_seconds(16000) # 1.0
```

Seconds are a convenience; the sample index is the source of truth.

```python
segment = sly.AudioSegment.from_seconds(tag_id, start_sec=1.0, end_sec=2.0, sample_rate=16000)
# -> start=16000, end=31999  (end_sec is exclusive, as people describe a span)
```

`channel` is a zero-based channel index, or `None` for the mixdown.

## Why the settings are stored

A spectrogram is not the audio — it is one of many possible transforms of it,
and the choice changes what is visible. Two tones 40 Hz apart are one line at
`fft_size=256` and two lines at `fft_size=1024`. A quiet event 70 dB down is
invisible at `min_db=-60` and obvious at `min_db=-100`. So the settings a label
was made under are part of the label.

```python
settings = sly.SpectrogramSettings(
    scale="mel",        # linear | log | mel
    fft_size=1024,      # power of two, 32..32768
    hop_length=256,
    window="hann",      # hann | hamming | blackman
    mel_bands=64,       # 2..512
    min_db=-100.0,
    max_db=0.0,
    channel=0,
)

api.audio.add_segment(
    project.id,
    recording.id,
    sly.AudioSegment(tag_id=tag_id, start=16000, end=31999, channel=0, settings=settings),
)
```

`settings=None` is meaningful rather than missing: it records that the label was
made on the waveform alone, by ear.

### Fingerprints

`settings.fingerprint` is a canonical hash over the fields that change the
numbers. `colormap` and `interpolation` only change how the array is painted, so
they are excluded — two labels made under the same analysis compare equal
regardless of palette.

```python
a.fingerprint == b.fingerprint   # same transform -> safe to merge or train together
```

One string answers "can these datasets be merged", "were these labels made under
the same view", and "does my training render match what the annotator saw".

## Reading back

```python
segments = api.audio.get_segments(recording_id)
for s in segments:
    print(s.start, s.end, s.channel, s.settings)
```

Both `entities.list` and `entities.info` omit `tags` and their `meta` from the
default projection, so `api.audio` asks for them explicitly
(`AudioApi.ENTITY_FIELDS`). `meta` is exactly where the settings live.

A segment records two channels, and they mean different things: `segment.channel`
is what the label is *about*, `segment.settings.channel` is what was on *screen*
when it was drawn. The labeling tool writes them independently, and so does the
SDK; neither overwrites the other.

## Local projects

`sly.download(api, project_id, dest_dir)` works for audio like any other
modality, and `sly.AudioProject` reads the result back:

```
project/
    meta.json
    ds0/
        audio/rain.wav
        ann/rain.wav.json
        audio_info/rain.wav.json     # only with save_audio_info=True
```

```python
sly.download(api, project_id, "/tmp/proj", save_audio_info=True)

project_fs = sly.AudioProject("/tmp/proj", sly.OpenMode.READ)
dataset_fs = project_fs.datasets.get("ds0")

ann = dataset_fs.get_ann("rain.wav", project_fs.meta)
ann.sample_rate, ann.sample_count, ann.channels   # the platform stores none of these
for segment in ann.tags:
    print(segment.name, segment.start, segment.end, segment.settings)

sly.upload_audio_project("/tmp/proj", api, workspace_id, "copy of engine-noise")
```

On disk a segment is identified by tag **name**, not by the server id, so a
downloaded project is readable without the server that issued the ids. The
recording's shape is stored in the annotation because the platform does not keep
it, and without it a sample range cannot be converted to seconds.

## Rendering

The platform deliberately stores no sample rate, duration or channel count for
audio — clients derive what they need — so the file is decoded locally.

```python
api.audio.download_path(recording_id, "/tmp/run1.wav")

info = sly.audio.get_audio_info("/tmp/run1.wav")   # header only, no decoding
samples, rate = sly.audio.read_audio("/tmp/run1.wav")

# the whole recording
spec = sly.audio.render_spectrogram(samples, rate, settings)

# just one labeled segment, for a training crop
crop = sly.audio.render_segment(samples, rate, segment.start, segment.end, segment.settings)

# raw power instead of clipped dB, for your own normalisation
raw = sly.audio.render_spectrogram(samples, rate, settings, as_db=False)
```

Rendering under `segment.settings` reproduces the analysis the annotator saw.
Decibels are **absolute** — `10*log10(power)`, calibrated so a full-scale sine
reads 0 dB, exactly as the tool paints them. Nothing is normalised to the loudest
point, which is what makes a crop, a full render and a second recording
comparable with each other.

### What the settings do and do not determine

Confirmed workflow: spectrogram settings are chosen **once per dataset or labeling
job** and not changed by individual annotators, and reproducing the view from the
stored parameters is sufficient — the rendered image is not stored. Per-label
settings may appear later, which is why the settings live on each label rather
than on the project.

* The **analysis** is fully determined by the stored settings. Every step is
  matched to the labeling tool: periodic windows, power scaled by
  `1/sum(window)^2` and doubled outside DC and Nyquist, frames centred on
  `k*hop_length`, HTK mel edges, the weighted-average mel projection, the
  frequency mapping, and the palette stops used by `to_image`.
* The **picture** also depends on the display height, which is *not* among the
  stored settings. Pass `rows=` to reproduce a particular on-screen grid:

  ```python
  spec = sly.audio.render_spectrogram(samples, rate, settings, rows=512)
  ```

  Without `rows`, you get the natural resolution — mel bands, or FFT bins.

`sly.audio.scale_position_to_hz(position, rate, settings)` converts a normalised
vertical position on the displayed spectrogram to the frequency the annotator saw,
using the tool's own mapping.

> Bit-exactness against the tool's WASM STFT is still unverified — that needs a
> golden-file fixture from the platform renderer, and float arithmetic will differ
> in the last places regardless. The analysis matches; the last bits are unproven.

WAV is decoded with the standard library. Other formats (FLAC, OGG, MP3, M4A)
need the optional `soundfile` package: `pip install supervisely[audio]`.

## Validation

Settings are validated on construction, against the **labeling tool's** bounds
rather than the API's. The API is currently looser — it accepts `mel_bands=1`
and `fft_size=65536`, which the toolbox then refuses to open, leaving a
recording that cannot be viewed. The SDK will not author those.

```python
sly.SpectrogramSettings(mel_bands=1)   # ValueError
sly.SpectrogramSettings(fft_size=1000) # ValueError -- not a power of two
```
