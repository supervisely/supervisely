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
```

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
    recording["id"],
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
segments = api.audio.get_segments(recording_id, dataset_id)
for s in segments:
    print(s.start, s.end, s.channel, s.settings)
```

`dataset_id` is required because the only API projection that returns the
settings is the list endpoint. `images.info` answers for a single recording but
silently drops `meta`, which is exactly where the settings live.

## Rendering

The platform stores no sample rate, duration or channel count for audio, so the
file has to be decoded locally.

```python
api.audio.download_path(recording_id, "/tmp/run1.wav")

info = sly.get_audio_info("/tmp/run1.wav")
samples, rate = sly.read_audio("/tmp/run1.wav")

# the whole recording
spec = sly.render_spectrogram(samples, rate, settings)

# just one labeled segment, for a training crop
crop = sly.render_segment(samples, rate, segment.start, segment.end, segment.settings)

# raw magnitude instead of clipped dB, for your own normalisation
raw = sly.render_spectrogram(samples, rate, settings, as_db=False)
```

Rendering under `segment.settings` reproduces the analysis the annotator saw.

> **Caveat.** The renderer matches the labeling tool *semantically* — same
> window, hop, frequency mapping and dB range. Whether it is bit-identical to
> the tool's WASM STFT has not been established. Treat the output as "the same
> analysis", not "the same bytes", until a golden-file comparison exists.

WAV is decoded with the standard library. Other formats (FLAC, OGG, MP3, M4A)
need the optional `soundfile` package.

## Validation

Settings are validated on construction, against the **labeling tool's** bounds
rather than the API's. The API is currently looser — it accepts `mel_bands=1`
and `fft_size=65536`, which the toolbox then refuses to open, leaving a
recording that cannot be viewed. The SDK will not author those.

```python
sly.SpectrogramSettings(mel_bands=1)   # ValueError
sly.SpectrogramSettings(fft_size=1000) # ValueError -- not a power of two
```
