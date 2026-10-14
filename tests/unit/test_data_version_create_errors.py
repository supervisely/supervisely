# coding: utf-8
"""DataVersion.create error handling around the commit.

Before the commit, a failure cancels the reservation and returns None. After the
commit the reservation no longer exists, so a failure in the preview step must
surface as itself rather than as the server's answer to a cancel request.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import requests

from supervisely.project.data_version import DataVersion
from supervisely.project.project_type import ProjectType


class PreviewError(Exception):
    pass


def _already_committed_error():
    response = requests.Response()
    response.status_code = 400
    response._content = b'{"message":"version is already committed"}'
    return requests.exceptions.HTTPError("400 Client Error", response=response)


def _version_api(project_type=ProjectType.IMAGES.value):
    dv = DataVersion(mock.MagicMock())
    project_info = SimpleNamespace(
        id=1, team_id=1, type=project_type, updated_at="2026-01-01T00:00:00Z"
    )
    dv._api.server_address = "https://dev.example.com"
    dv.initialize = mock.MagicMock()
    dv.versions = {"1": {"number": 1}, "latest": 1}
    dv._generate_save_path = mock.MagicMock(return_value="/system/versions/1/2.tar.zst")
    dv._get_latest_id = mock.MagicMock(return_value=1)
    dv.reserve = mock.MagicMock(return_value=(2, "token"))
    dv._compress_and_upload = mock.MagicMock(return_value=SimpleNamespace(id=10))
    dv.set_map = mock.MagicMock()
    dv.commit = mock.MagicMock()
    dv.cancel_reservation = mock.MagicMock(side_effect=_already_committed_error())
    dv.enable_preview = mock.MagicMock(side_effect=PreviewError("preview failed"))
    return dv, project_info


def test_preview_failure_after_commit_raises_the_preview_error():
    dv, project_info = _version_api()

    with pytest.raises(PreviewError, match="preview failed"):
        dv.create(project_info, "v2", enable_preview=True)

    dv.commit.assert_called_once()
    dv.cancel_reservation.assert_not_called()


def test_preview_failure_after_commit_keeps_the_version_in_the_map():
    dv, project_info = _version_api()

    with pytest.raises(PreviewError):
        dv.create(project_info, "v2", enable_preview=True)

    assert dv.versions["latest"] == 2
    assert dv.versions["2"]["number"] == 2
    assert "preview" not in dv.versions["2"]


def test_failure_before_commit_still_cancels_the_reservation():
    dv, project_info = _version_api()
    dv.cancel_reservation = mock.MagicMock(return_value=True)
    dv.commit = mock.MagicMock(side_effect=RuntimeError("Failed to commit version"))

    assert dv.create(project_info, "v2", enable_preview=True) is None
    dv.cancel_reservation.assert_called_once_with(2, "token")
    dv.enable_preview.assert_not_called()


def test_successful_preview_returns_the_version_id():
    dv, project_info = _version_api()
    dv.enable_preview = mock.MagicMock(return_value=SimpleNamespace(id=99))

    assert dv.create(project_info, "v2", enable_preview=True) == 2
    assert dv.versions["2"]["preview"] == 99
    dv.cancel_reservation.assert_not_called()


def test_preview_failure_logs_the_version_id_and_how_to_rebuild_the_preview():
    dv, project_info = _version_api()
    # The local map would number this version 8, whatever the server says.
    dv.versions = {"1": {"number": 7}, "latest": 1}
    dv.get_info_by_id = mock.MagicMock(side_effect=AssertionError("no request on the error path"))

    with mock.patch("supervisely.project.data_version.logger") as log:
        with pytest.raises(PreviewError, match="preview failed"):
            dv.create(project_info, "v2", enable_preview=True)

    message = log.error.call_args[0][0]
    assert message.startswith("Version with ID 2 was created")
    assert "api.project.version.enable_preview(1, 2, overwrite=True)" in message
    assert "PreviewError" in message


def test_map_write_failure_after_preview_raises_its_own_error():
    dv, project_info = _version_api()
    dv.enable_preview = mock.MagicMock(return_value=SimpleNamespace(id=99))
    dv.set_map = mock.MagicMock(side_effect=[None, RuntimeError("Failed to save versions")])

    with pytest.raises(RuntimeError, match="Failed to save versions"):
        dv.create(project_info, "v2", enable_preview=True)
    dv.cancel_reservation.assert_not_called()
