"""
Regression test for ConfusionMatrix._calculate_totals aliasing bug.

`_calculate_totals()` built `matrix_data.classes` by passing
`self._parsed_data["classes"]` (the same list object backing
`table_data.classes`) straight into `get_unpacked_data()`, which assigns it
to the output dict without copying. `table_data.classes` and
`matrix_data.classes` ended up being the literal same list object.

DataJson/StateJson sync computes a JSON patch between the previous and
current widget state and applies it via list index operations (add/remove at
a specific position). Diffing `table_data.classes` and `matrix_data.classes`
independently produces two SEPARATE sets of index-based ops for what the
diff engine believes are two independent lists. Applying the first set
mutates the (aliased) list in place; the second set then operates on a list
that has already shrunk/grown, referencing indices that no longer exist —
crashing with `JsonPatchConflict: can't remove a non-existent object`.

This reproduces the exact failure: read_pandas() with a shrinking class
count, then send_changes(), matching the widget's real usage where any
button/handler triggers a sync after mutating the DataJson).
"""

import copy

import pandas as pd
import pytest

from supervisely.app.content import DataJson
from supervisely.app.widgets import ConfusionMatrix


def _matrix(n, seed_multiplier):
    labels = [f"cls_{i}" for i in range(n)]
    data = [
        [round((r * c * seed_multiplier) % 15, 1) for c in range(1, n + 1)]
        for r in range(1, n + 1)
    ]
    return pd.DataFrame(data=data, index=labels, columns=labels)


@pytest.fixture
def data_json():
    d = DataJson()
    d.clear()
    d._last = {}
    yield d
    d.clear()
    d._last = {}


def test_table_data_and_matrix_data_classes_are_independent_lists(data_json):
    """table_data.classes and matrix_data.classes must not be the same list
    object -- otherwise mutating one via a JSON patch corrupts the other."""
    cm = ConfusionMatrix()
    cm.read_pandas(_matrix(5, 0.37))

    wid = cm.widget_id
    table_classes = data_json[wid]["table_data"]["classes"]
    matrix_classes = data_json[wid]["matrix_data"]["classes"]

    assert table_classes is not matrix_classes
    assert table_classes == matrix_classes


def test_read_pandas_shrinking_classes_survives_send_changes(data_json):
    """The exact crash scenario: read_pandas() to a smaller class count,
    then send_changes() (as any real button/handler would trigger), must not
    raise JsonPatchConflict."""
    cm = ConfusionMatrix()
    cm.read_pandas(_matrix(10, 0.37))
    data_json.send_changes()

    for i in range(20):
        n = 6 if i % 2 == 0 else 10
        cm.read_pandas(_matrix(n, 0.37 if i % 2 == 0 else 0.53))
        data_json.send_changes()  # must not raise


def test_matrix_data_classes_matches_table_data_after_resize(data_json):
    """After shrinking and re-growing, both class lists must still agree."""
    cm = ConfusionMatrix()
    cm.read_pandas(_matrix(10, 0.37))
    data_json.send_changes()

    cm.read_pandas(_matrix(6, 0.53))
    data_json.send_changes()

    wid = cm.widget_id
    assert data_json[wid]["table_data"]["classes"] == data_json[wid]["matrix_data"]["classes"]
    assert len(data_json[wid]["table_data"]["classes"]) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
