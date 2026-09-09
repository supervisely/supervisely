import json

from click.testing import CliRunner

from supervisely.api.app_api import _context_menu_targets
from supervisely.cli.app_commands import _CONTEXT_TARGETS, app_group
from supervisely.cli.common import CliState


class FakeModuleInfo:
    def __init__(self, config=None, module_id=81, slug="owner/example"):
        self.id = module_id
        self.slug = slug
        self.name = "Example app"
        self.type = "app"
        self.config = config or {}
        self.meta = {"releases": [{"version": "v2.0.0"}, {"version": "v1.0.0"}]}

    def get_latest_release(self):
        return self.meta["releases"][0]

    def get_modal_window_arguments(self):
        return self.config.get("modalTemplateState", {})

    def get_context_menu_targets(self):
        context_menu = self.config.get("context_menu", self.config.get("contextMenu", {}))
        return context_menu.get("target", [])


class FakeAppApi:
    def __init__(self, module_info=None):
        self.module_info = module_info or FakeModuleInfo()
        self.module_info_calls = []
        self.list_calls = []
        self.session_calls = []
        self.start_calls = []
        self.stop_calls = []

    def get_ecosystem_module_info(self, *, slug, version=None):
        self.module_info_calls.append({"slug": slug, "version": version})
        return self.module_info

    def get_list_ecosystem_modules(self, *, search=None):
        self.list_calls.append(search)
        return [
            {
                "id": 81,
                "slug": "owner/example",
                "name": "Example app",
                "type": "app",
                "meta": {"releases": [{"version": "v2.0.0"}]},
            }
        ]

    def get_sessions(self, **kwargs):
        self.session_calls.append(kwargs)
        return [
            {
                "task_id": 12,
                "user_id": 3,
                "module_id": 81,
                "app_id": 7,
                "details": {"status": "started"},
            }
        ]

    def start(self, **kwargs):
        self.start_calls.append(kwargs)
        return {"task_id": 91, "module_id": kwargs["module_id"]}

    def stop(self, task_id):
        self.stop_calls.append(task_id)
        return "stopped"


class FakeApi:
    def __init__(self, module_info=None):
        self.app = FakeAppApi(module_info)


def invoke_json(fake_api, arguments):
    return CliRunner().invoke(
        app_group,
        arguments,
        obj=CliState(api=fake_api, json_output=True),
    )


def test_cli_context_target_metadata_matches_sdk_argument_builder():
    assert _CONTEXT_TARGETS.keys() == _context_menu_targets.keys()
    for target, sdk_metadata in _context_menu_targets.items():
        cli_metadata = _CONTEXT_TARGETS[target]
        assert cli_metadata["key"] == sdk_metadata["key"]
        assert cli_metadata["help"] == sdk_metadata["help"]
        assert cli_metadata.get("type") == sdk_metadata.get("type")


def test_list_filters_and_summarizes_modules():
    api = FakeApi()

    result = invoke_json(api, ["list", "--search", "export"])

    assert result.exit_code == 0, result.output
    assert api.app.list_calls == ["export"]
    assert json.loads(result.output) == [
        {
            "id": 81,
            "slug": "owner/example",
            "name": "Example app",
            "type": "app",
            "releases": ["v2.0.0"],
        }
    ]


def test_params_resolves_branch_specific_configuration():
    module = FakeModuleInfo(
        config={
            "modalTemplateState": {"confidence": 0.75},
            "contextMenu": {"target": ["ecosystem"]},
        }
    )
    api = FakeApi(module)

    result = invoke_json(
        api, ["params", "--slug", "owner/example", "--branch", "feature/cli"]
    )

    assert result.exit_code == 0, result.output
    assert api.app.module_info_calls == [
        {"slug": "owner/example", "version": "feature/cli"}
    ]
    payload = json.loads(result.output)
    assert payload["module_id"] == 81
    assert payload["selected_version"] == "feature/cli"
    assert payload["is_branch"] is True
    assert payload["modal_template_state"] == {"confidence": 0.75}
    assert payload["params_template"] == {"confidence": 0.75}
    assert payload["can_run_from_ecosystem"] is True


def test_params_explains_required_context_target():
    module = FakeModuleInfo(
        config={
            "modalTemplateState": {"include_annotations": True},
            "contextMenu": {"target": ["images_project"]},
        }
    )
    api = FakeApi(module)

    result = invoke_json(api, ["params", "--slug", "owner/example"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["can_run_from_ecosystem"] is False
    assert payload["required_context_target"] == {
        "target": "images_project",
        "type": "integer",
        "key": "slyProjectId",
        "help": "Context menu of images project. Target value is project id.",
        "placeholder": "<integer>",
    }
    assert payload["params_template"] == {
        "include_annotations": True,
        "slyProjectId": "<integer>",
    }
    assert "cannot be inferred" in payload["limitations"][0]


def test_describe_version_and_branch_are_mutually_exclusive():
    api = FakeApi()

    result = invoke_json(
        api,
        [
            "describe",
            "--slug",
            "owner/example",
            "--branch",
            "main",
            "--version",
            "v1.0.0",
        ],
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output
    assert api.app.module_info_calls == []


def test_run_preserves_arbitrary_params_and_branch_settings(tmp_path):
    module = FakeModuleInfo(
        config={
            "modalTemplateState": {"threshold": 0.5},
            "contextMenu": {"target": ["images_project"]},
        }
    )
    api = FakeApi(module)
    params = {
        "threshold": 0.9,
        "slyProjectId": 123,
        "custom": {"nested": [1, 2, 3]},
    }
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    result = invoke_json(
        api,
        [
            "run",
            "--slug",
            "owner/example",
            "--workspace-id",
            "8",
            "--agent-id",
            "4",
            "--params-file",
            str(params_path),
            "--branch",
            "feature/cli",
            "--task-name",
            "CLI task",
            "--description",
            "started by an agent",
            "--log-level",
            "debug",
            "--restart-policy",
            "on_error",
        ],
    )

    assert result.exit_code == 0, result.output
    assert api.app.module_info_calls == [
        {"slug": "owner/example", "version": "feature/cli"}
    ]
    assert api.app.start_calls == [
        {
            "agent_id": 4,
            "module_id": 81,
            "workspace_id": 8,
            "description": "started by an agent",
            "params": params,
            "log_level": "debug",
            "app_version": "feature/cli",
            "is_branch": True,
            "task_name": "CLI task",
            "restart_policy": "on_error",
        }
    ]
    assert json.loads(result.output) == {"module_id": 81, "task_id": 91}


def test_run_selects_release_without_branch_mode():
    module = FakeModuleInfo(config={"modalTemplateState": {"confidence": 0.6}})
    api = FakeApi(module)

    result = invoke_json(
        api,
        [
            "run",
            "--slug",
            "owner/example",
            "--workspace-id",
            "8",
            "--version",
            "v1.0.0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert api.app.module_info_calls == [
        {"slug": "owner/example", "version": "v1.0.0"}
    ]
    assert api.app.start_calls[0]["app_version"] == "v1.0.0"
    assert api.app.start_calls[0]["is_branch"] is False
    assert api.app.start_calls[0]["params"] == {"confidence": 0.6}


def test_run_rejects_invalid_and_non_object_parameter_files(tmp_path):
    api = FakeApi()
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{", encoding="utf-8")
    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")

    invalid = invoke_json(
        api,
        [
            "run",
            "--slug",
            "owner/example",
            "--workspace-id",
            "8",
            "--params-file",
            str(invalid_path),
        ],
    )
    non_object = invoke_json(
        api,
        [
            "run",
            "--slug",
            "owner/example",
            "--workspace-id",
            "8",
            "--params-file",
            str(list_path),
        ],
    )

    assert invalid.exit_code == 1
    assert "Invalid JSON" in invalid.output
    assert non_object.exit_code == 1
    assert "must contain an object" in non_object.output
    assert api.app.start_calls == []


def test_run_validates_context_target_inside_wrapped_state(tmp_path):
    module = FakeModuleInfo(
        config={"contextMenu": {"target": ["images_dataset"]}}
    )
    api = FakeApi(module)
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"state": {"slyDatasetId": "42"}}), encoding="utf-8")

    result = invoke_json(
        api,
        [
            "run",
            "--slug",
            "owner/example",
            "--workspace-id",
            "8",
            "--params-file",
            str(params_path),
        ],
    )

    assert result.exit_code == 1
    assert "slyDatasetId' must be integer, got str" in result.output
    assert api.app.start_calls == []


def test_run_accepts_top_level_context_target_beside_state(tmp_path):
    module = FakeModuleInfo(
        config={"contextMenu": {"target": ["images_project"]}}
    )
    api = FakeApi(module)
    params = {"state": {"confidence": 0.8}, "slyProjectId": 42}
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    result = invoke_json(
        api,
        [
            "run",
            "--slug",
            "owner/example",
            "--workspace-id",
            "8",
            "--params-file",
            str(params_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert api.app.start_calls[0]["params"] == params


def test_run_requires_params_for_context_only_app():
    module = FakeModuleInfo(config={"contextMenu": {"target": ["files_file"]}})
    api = FakeApi(module)

    result = invoke_json(
        api, ["run", "--slug", "owner/example", "--workspace-id", "8"]
    )

    assert result.exit_code == 1
    assert "must be launched from a context target" in result.output
    assert api.app.start_calls == []


def test_sessions_resolves_slug_and_passes_repeatable_statuses():
    api = FakeApi()

    result = invoke_json(
        api,
        [
            "sessions",
            "--team-id",
            "9",
            "--slug",
            "owner/example",
            "--status",
            "started",
            "--status",
            "queued",
            "--show-disabled",
            "--with-shared",
        ],
    )

    assert result.exit_code == 0, result.output
    assert api.app.session_calls == [
        {
            "team_id": 9,
            "module_id": 81,
            "statuses": ["started", "queued"],
            "show_disabled": True,
            "with_shared": True,
        }
    ]


def test_sessions_rejects_unknown_status_before_calling_api():
    api = FakeApi()

    result = invoke_json(
        api,
        [
            "sessions",
            "--team-id",
            "9",
            "--slug",
            "owner/example",
            "--status",
            "unknown",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--status'" in result.output
    assert api.app.module_info_calls == []


def test_stop_requires_yes_before_calling_api():
    api = FakeApi()

    denied = invoke_json(api, ["stop", "--task-id", "91"])
    allowed = invoke_json(api, ["stop", "--task-id", "91", "--yes"])

    assert denied.exit_code == 2
    assert "requires --yes" in denied.output
    assert allowed.exit_code == 0, allowed.output
    assert api.app.stop_calls == [91]
    assert json.loads(allowed.output) == {"status": "stopped", "task_id": 91}
