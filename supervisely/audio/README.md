# Audio modality

Audio projects hold recordings labeled with **time segments**. The spectrogram
they are labeled against is configured once for the whole project, so the picture
every annotator was looking at can be reproduced later and fed to training.

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

## The spectrogram belongs to the project

A spectrogram is not the audio — it is one of many possible transforms of it,
and the choice changes what is visible. Two tones 40 Hz apart are one line at
`fft_size=256` and two lines at `fft_size=1024`. A quiet event 70 dB down is
invisible at `min_db=-60` and obvious at `min_db=-100`. So every recording in a
project has to be analysed the same way, or two annotators are not looking at
the same evidence.

The platform stores the settings in `projects.settings.spectrogram` and applies
them to every recording. Set them once, at the start of the project.

```python
settings = sly.SpectrogramSettings(
    scale="mel",            # linear | log | mel
    fft_size=1024,          # power of two, 32..32768
    hop_length=256,
    window="hann",          # hann | hamming | blackman
    mel_bands=64,           # 2..512
    min_db=-100.0,
    max_db=0.0,
    colormap="magma",       # viridis | magma | grayscale
    interpolation="sharp",  # sharp | smooth
)

api.audio.set_spectrogram_settings(project.id, settings)
api.audio.get_spectrogram_settings(project.id)   # the defaults, if never configured
```

Changing them is a project-wide change that everyone sees, and it needs
permission to edit the project (`PROJECTS.UPDATE`) rather than permission to
label in it — an annotator cannot retune the analysis mid-job. Existing labels
are left untouched, which is the reason to set it once and leave it.

`channel` is not one of the settings: which channel is on screen is navigation,
so it is an argument to the render functions instead.

```python
api.audio.add_segment(
    project.id,
    recording.id,
    sly.AudioSegment(tag_id=tag_id, start=16000, end=31999, channel=0),
)
```

### Fingerprints

`settings.fingerprint` is a canonical hash over the fields that change the
numbers. `colormap` and `interpolation` only change how the array is painted, so
they are excluded — two projects analysing identically compare equal regardless
of palette.

```python
a.fingerprint == b.fingerprint   # same transform -> safe to merge or train together
```

One string answers "can these datasets be merged", "were these labels made under
the same analysis", and "does my training render match what the annotator saw".

## Reading back

```python
segments = api.audio.get_segments(recording_id)
for s in segments:
    print(s.start, s.end, s.channel, s.meta)
```

Both `entities.list` and `entities.info` omit `tags` and their `meta` from the
default projection, so `api.audio` asks for them explicitly
(`AudioApi.ENTITY_FIELDS`). `meta` is where a segment's channel lives.

`meta` is a free-form options object shared with other clients, so keys this SDK
does not know about are read back and written out unchanged — including the
`spectrogram` object that labels made before the settings moved onto the project
still carry. It is history, not configuration: the project's settings are what
the tool renders with.

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
    print(segment.name, segment.start, segment.end, segment.channel)

# the project's analysis travels in meta.json and is restored on upload
sly.SpectrogramSettings.from_json(project_fs.meta.project_settings.spectrogram)

sly.upload_audio_project("/tmp/proj", api, workspace_id, "copy of engine-noise")
```

On disk a segment is identified by tag **name**, not by the server id, so a
downloaded project is readable without the server that issued the ids. The
recording's shape is stored in the annotation because the platform does not keep
it, and without it a sample range cannot be converted to seconds.

### Import into an existing project

`sly.ImportManager` is what the Auto Import app runs, so it works inside an
app, in the app's data directory (`SLY_APP_DATA_DIR`). It accepts two inputs for
an audio project: recordings in any folder structure, uploaded without labels, or
a project in the layout above, uploaded with its segments. Segments are matched
to the destination's tags by name; a conflicting tag is renamed, as for the
other modalities.

```python
importer = sly.ImportManager("/tmp/proj", sly.ProjectType.AUDIO)
importer.upload_dataset(dataset_id)
```

The spectrogram settings in `meta.json` are applied only when the destination
project is not configured yet and holds no recordings. A project that already
has settings, or labels drawn under the defaults, keeps its own, and a mismatch
is logged: changing them would change what the existing labels mean. Audio is
never added by link; the files are always uploaded.

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
settings = api.audio.get_spectrogram_settings(project.id)
crop = sly.audio.render_segment(
    samples, rate, segment.start, segment.end, settings, channel=segment.channel
)

# raw power instead of clipped dB, for your own normalisation
raw = sly.audio.render_spectrogram(samples, rate, settings, as_db=False)
```

Rendering under the project's settings reproduces the analysis the annotator saw.
Decibels are **absolute** — `10*log10(power)`, calibrated so a full-scale sine
reads 0 dB, exactly as the tool paints them. Nothing is normalised to the loudest
point, which is what makes a crop, a full render and a second recording
comparable with each other.

### What the settings do and do not determine

Confirmed workflow: spectrogram settings are chosen **once at the start of a
project** and not changed afterwards; only the audio is stored, never a rendered
spectrogram image; and the spectrogram is regenerated from the stored samples in
whatever framework does the training. So the stored parameters are the whole
contract — there is no image to fall back on.

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

> Bit-exactness against the tool's WASM STFT is not a goal and is not claimed.
> The training input is rendered by the consumer's own framework, not by the
> toolbox, so what has to hold is that the stored parameters pin the analysis
> unambiguously — and float arithmetic differs in the last places between any
> two implementations anyway.

### Reproducing the analysis in PyTorch or TensorFlow

The settings carry enough to rebuild the same analysis outside this SDK, which
is what a training pipeline usually wants. Five conventions are not implied by
the field names, and getting any of them wrong is a visible error rather than a
rounding one:

| | What the settings mean |
|---|---|
| Window | **Periodic** (`phase = 2*pi*i/N`), not the symmetric `numpy.hanning` family |
| Padding | Frames **centred** on `k*hop_length`, padded with **zeros** — not reflected |
| Scaling | Power `/sum(window)^2`, then `x4` on every bin except DC and Nyquist |
| Mel | **HTK** edges from 0 to Nyquist, and a weighted **average** — divide by the filterbank column sums, do not take the weighted sum |
| dB | `10*log10(power)` **absolute**, floor `1e-20`, clipped to `[min_db, max_db]`. No `top_db`, no per-render max |

```python
# periodic_window() builds the named window as `a0 - a1*cos(p) + a2*cos(2p)`,
# p = 2*pi*i/N -- see the test module below for the coefficients.
win = torch.from_numpy(periodic_window(settings.window, settings.fft_size))
spec = torch.stft(samples, n_fft=settings.fft_size, hop_length=settings.hop_length,
                  win_length=settings.fft_size, window=win,
                  center=True, pad_mode="constant",   # torch defaults to "reflect"
                  normalized=False, return_complex=True)
power = spec.abs().double() ** 2 / float(win.double().sum()) ** 2
power[1:-1] *= 4.0

fb = torchaudio.functional.melscale_fbanks(settings.fft_size // 2 + 1, 0.0, rate / 2,
                                           settings.mel_bands, rate,
                                           norm=None, mel_scale="htk")
power = (fb.T.double() @ power) / fb.sum(0).double()[:, None]     # average, not sum
db = (10 * torch.log10(power.clamp_min(1e-20))).clamp(settings.min_db, settings.max_db)
```

`tests/audio_tests/test_framework_parity.py` runs this and the `tf.signal`
equivalent against `render_spectrogram` and is the executable version of the
table. Both agree to under 0.07 dB across scales, windows and FFT sizes — the
residual is float32 versus float64, and a 100 dB display range quantised to 256
levels is 0.4 dB per level, so it is not something an annotator could see. The
tests skip themselves when the framework is not installed; neither is a
dependency of this SDK.

WAV is decoded with the standard library. Other formats (FLAC, OGG, MP3, M4A)
need the optional `soundfile` package: `pip install supervisely[audio]`.

## Validation

Settings are validated on construction, against the bounds the API enforces on
`projects.settings.spectrogram`: an enumerated set of FFT sizes, 2..512 mel
bands, a hop of at least one sample, and a dB range the right way round. The
labeling tool's bounds are the same ones, so a value this SDK accepts is a value
the toolbox can open. Checking locally turns a 400 into a specific error.

```python
sly.SpectrogramSettings(mel_bands=1)   # ValueError
sly.SpectrogramSettings(fft_size=1000) # ValueError -- not a power of two
```
