"""
Tests for Select widget value handling.

Regression: in multiple mode the frontend keeps the value as an array and calls
.push() on it when an option is clicked. Clearing the selection with
set_value("") used to store a scalar string, which broke option clicks until a
page reload. set_value must coerce scalars to a list in multiple mode.
"""

from supervisely.app.widgets import Select


def _multiple_select():
    return Select(
        items=[Select.Item("a"), Select.Item("b")],
        multiple=True,
    )


def test_multiple_set_value_empty_string_becomes_list():
    select = _multiple_select()
    select.set_value("")  # common "clear selection" idiom
    assert select.get_value() == []


def test_multiple_set_value_none_becomes_list():
    select = _multiple_select()
    select.set_value(None)
    assert select.get_value() == []


def test_multiple_set_value_scalar_wrapped_in_list():
    select = _multiple_select()
    select.set_value("a")
    assert select.get_value() == ["a"]


def test_multiple_set_value_list_preserved():
    select = _multiple_select()
    select.set_value(["a", "b"])
    assert select.get_value() == ["a", "b"]


def test_single_set_value_empty_string_unchanged():
    select = Select(items=[Select.Item("a")], multiple=False)
    select.set_value("")
    assert select.get_value() == ""
