import json
from collections import namedtuple

import click
from click.testing import CliRunner

from supervisely.cli.common import (
    CliState,
    emit,
    get_api,
    read_json_object,
    to_jsonable,
)


def test_to_jsonable_redacts_credentials_recursively():
    AgentInfo = namedtuple("AgentInfo", ["id", "token", "capabilities"])
    value = AgentInfo(
        id=7,
        token="top-secret",
        capabilities={"authorization": "nested-secret", "gpu": True},
    )

    assert to_jsonable(value) == {
        "id": 7,
        "token": "***",
        "capabilities": {"authorization": "***", "gpu": True},
    }


def test_emit_json_is_machine_readable():
    @click.command()
    def command():
        emit({"id": 12, "name": "example"})

    result = CliRunner().invoke(command, obj=CliState(json_output=True))

    assert result.exit_code == 0
    assert json.loads(result.output) == {"id": 12, "name": "example"}


def test_read_json_object_rejects_non_object(tmp_path):
    path = tmp_path / "params.json"
    path.write_text("[]", encoding="utf-8")

    try:
        read_json_object(str(path))
    except click.ClickException as exc:
        assert "must contain an object" in str(exc)
    else:
        raise AssertionError("Expected a ClickException")


def test_get_api_combines_explicit_and_environment_credentials(monkeypatch):
    import supervisely as sly

    class FakeApi:
        created = []

        def __init__(self, server_address, token):
            self.server_address = server_address
            self.token = token
            self.created.append((server_address, token))

        @classmethod
        def from_env(cls):
            raise AssertionError("Partial overrides must not require complete env credentials")

    monkeypatch.setattr(sly, "Api", FakeApi)
    monkeypatch.setenv("API_TOKEN", "environment-token")

    @click.command()
    def command():
        api = get_api()
        emit({"server_address": api.server_address})

    state = CliState(
        json_output=True,
        server_address="https://override.example",
    )
    result = CliRunner().invoke(command, obj=state)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "server_address": "https://override.example"
    }
    assert FakeApi.created == [
        ("https://override.example", "environment-token")
    ]
    assert state.api.server_address == "https://override.example"


def test_get_api_uses_complete_explicit_credentials_without_environment(monkeypatch):
    import supervisely as sly

    class FakeApi:
        created = []

        def __init__(self, server_address, token):
            self.server_address = server_address
            self.token = token
            self.created.append((server_address, token))

        @classmethod
        def from_env(cls):
            raise AssertionError("from_env must not be used with complete credentials")

    monkeypatch.setattr(sly, "Api", FakeApi)

    @click.command()
    def command():
        get_api()

    state = CliState(
        server_address="https://explicit.example",
        api_token="explicit-token",
    )
    result = CliRunner().invoke(command, obj=state)

    assert result.exit_code == 0, result.output
    assert FakeApi.created == [
        ("https://explicit.example", "explicit-token")
    ]
