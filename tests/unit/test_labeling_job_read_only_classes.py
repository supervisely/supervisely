"""Read-only classes in labeling jobs.

``api.labeling_job.create(read_only_classes=...)`` sends ``meta.readOnlyClasses``. ``jobs.info``
returns it as ``[{"id", "name"}]`` and ``jobs.list`` as ``[{"id", "title"}]``;
``LabelingJobInfo.read_only_classes`` holds the names, so ``clone`` carries them into the new job.
Instances that predate the setting omit it, and the field is then ``None``.
"""

from unittest.mock import MagicMock

from supervisely.api.labeling_job_api import LabelingJobApi


def _job_json(meta_extra=None):
    meta = {
        "classes": [{"id": 1, "name": "vehicle"}],
        "projectTags": [],
        "imageTags": [],
        "imageFiguresLimit": 0,
        "imageTagsLimit": 0,
        "range": {"start": None, "end": None},
    }
    meta.update(meta_extra or {})
    return {
        "id": 7,
        "name": "job",
        "readme": "",
        "description": "",
        "teamId": 1,
        "datasetId": 3,
        "userId": 5,
        "reviewerId": 5,
        "entities": [{"id": 11}],
        "meta": meta,
    }


def _api(job_json):
    api = MagicMock()

    def post(method, data):
        response = MagicMock()
        if method == "jobs.add":
            response.json.return_value = [{"id": 7}]
        elif method == "jobs.info":
            response.json.return_value = job_json
        return response

    api.post.side_effect = post
    return api


def _sent_meta(api):
    calls = [c for c in api.post.call_args_list if c.args[0] == "jobs.add"]
    assert len(calls) == 1
    return calls[0].args[1]["meta"]


def test_create_sends_read_only_classes():
    api = _api(_job_json())
    LabelingJobApi(api).create(
        "job", 3, [5], classes_to_label=["vehicle"], read_only_classes=["hint_box"]
    )
    assert _sent_meta(api)["readOnlyClasses"] == ["hint_box"]


def test_create_without_read_only_classes_omits_the_key():
    api = _api(_job_json())
    LabelingJobApi(api).create("job", 3, [5], classes_to_label=["vehicle"])
    assert "readOnlyClasses" not in _sent_meta(api)


def test_info_holds_read_only_class_names():
    job_api = LabelingJobApi(_api(None))
    info = job_api._convert_json_info(
        _job_json({"readOnlyClasses": [{"id": 2, "name": "hint_box"}]})
    )
    assert info.read_only_classes == ["hint_box"]
    assert info.classes_to_label == ["vehicle"]


def test_info_reads_jobs_list_title_shape():
    info = LabelingJobApi(_api(None))._convert_json_info(
        _job_json({"readOnlyClasses": [{"id": 2, "title": "hint_box"}]})
    )
    assert info.read_only_classes == ["hint_box"]


def test_info_is_none_when_server_omits_read_only_classes():
    info = LabelingJobApi(_api(None))._convert_json_info(_job_json())
    assert info.read_only_classes is None


def test_clone_carries_read_only_classes():
    api = _api(_job_json({"readOnlyClasses": [{"id": 2, "name": "hint_box"}]}))
    LabelingJobApi(api).clone(7, new_title="copy")
    assert _sent_meta(api)["readOnlyClasses"] == ["hint_box"]


def test_clone_from_server_without_setting_omits_the_key():
    api = _api(_job_json())
    LabelingJobApi(api).clone(7, new_title="copy")
    assert "readOnlyClasses" not in _sent_meta(api)
