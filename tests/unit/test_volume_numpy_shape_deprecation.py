"""NumPy 2.5 deprecates setting ``ndarray.shape``, which SimpleITK and scikit-image still do.

The SDK's volume reads and mesh export must work when DeprecationWarnings are errors, and
silencing it must not leak into the caller's warning filters, from one thread or several.
"""

import os
import threading
import time
import warnings

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from supervisely.geometry.mask_3d import Mask3D
from supervisely.volume import volume as sly_volume

needs_numpy_2_5 = pytest.mark.skipif(
    np.lib.NumpyVersion(np.__version__) < "2.5.0.dev0",
    reason="setting ndarray.shape is deprecated from NumPy 2.5",
)


@pytest.fixture
def nrrd_path(tmp_path):
    data = np.arange(6 * 8 * 10, dtype=np.int16).reshape(6, 8, 10)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((0.5, 0.75, 2.0))
    path = str(tmp_path / "volume.nrrd")
    sitk.WriteImage(image, path)
    return path


def _cube_mask():
    data = np.zeros((12, 12, 12), dtype=np.bool_)
    data[3:9, 2:10, 4:8] = True
    return data


def _call_volume_paths(nrrd_path, with_mesh=True):
    sly_volume.read_nrrd_serie_volume_np(nrrd_path)
    sly_volume.convert_3d_nifti_to_nrrd(nrrd_path)
    mask = Mask3D(
        _cube_mask(),
        volume_header={
            "space": "left-posterior-superior",
            "space origin": [1.0, 2.0, 3.0],
            "space directions": [[0, 0.7, 0], [0.8, 0, 0], [0, 0, 2.5]],
        },
    )
    mask.orient_ras()
    if with_mesh:
        sly_volume.convert_3d_geometry_to_mesh(Mask3D(_cube_mask()), spacing=(1.0, 1.0, 1.0))


def _has_skimage():
    try:
        import skimage  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        return False
    return True


@needs_numpy_2_5
def test_volume_paths_work_with_deprecation_warnings_as_errors(nrrd_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _call_volume_paths(nrrd_path, with_mesh=_has_skimage())


def test_caller_filters_are_unchanged_and_own_warning_still_raised(nrrd_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        filters_before = list(warnings.filters)
        try:
            _call_volume_paths(nrrd_path, with_mesh=_has_skimage())
        except DeprecationWarning:
            pass
        assert warnings.filters == filters_before
        with pytest.raises(DeprecationWarning):
            warnings.warn("Setting the shape on a NumPy array", DeprecationWarning)


@needs_numpy_2_5
def test_overlapping_calls_from_threads_do_not_leave_the_filter_installed():
    filters_before = list(warnings.filters)
    first_inside = threading.Event()
    second_started = threading.Event()
    errors = []

    def first():
        try:
            with sly_volume._ignore_numpy_shape_deprecation():
                first_inside.set()
                second_started.wait(5)
                time.sleep(0.2)  # give the second thread time to enter if nothing stops it
        except BaseException as e:  # pylint: disable=broad-except
            errors.append(e)
            first_inside.set()

    def second():
        try:
            first_inside.wait(5)
            second_started.set()
            with sly_volume._ignore_numpy_shape_deprecation():
                time.sleep(0.4)  # outlive the first thread's exit
        except BaseException as e:  # pylint: disable=broad-except
            errors.append(e)

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert warnings.filters == filters_before


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_child_forked_while_filter_is_held_can_still_read(nrrd_path):
    inside = threading.Event()
    release = threading.Event()

    def hold():
        with sly_volume._ignore_numpy_shape_deprecation():
            inside.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    inside.wait(5)
    try:
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                sly_volume.read_nrrd_serie_volume_np(nrrd_path)
                code = 0
            finally:
                os._exit(code)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            time.sleep(0.1)
        else:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            pytest.fail("forked child hung reading a volume")
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    finally:
        release.set()
        holder.join(10)
