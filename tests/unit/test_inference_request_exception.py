from supervisely.nn.inference.inference_request import InferenceRequest


def _fail_in_model():
    raise ValueError("Item pexels-photo-744487.png not found in the project.")


def test_exception_json_keeps_traceback_of_the_raised_exception():
    inference_request = InferenceRequest()
    try:
        _fail_in_model()
    except ValueError as exc:
        inference_request.exception = exc

    # Serialized later, outside the except block, as the HTTP handler does
    exception_json = inference_request.exception_json()

    assert exception_json["type"] == "ValueError"
    assert "_fail_in_model" in exception_json["traceback"]
    assert "NoneType: None" not in exception_json["traceback"]
    assert inference_request.to_json()["stage"] == InferenceRequest.Stage.ERROR
