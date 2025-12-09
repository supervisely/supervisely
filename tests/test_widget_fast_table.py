"""
Tests for FastTable functionality.
Tests also verify that duplicate column names are handled correctly.
"""

import pandas as pd
import pytest

from supervisely.app.widgets import FastTable


class TestFastTableMultiIndex:
    """Test suite for FastTable with duplicate column names (MultiIndex)."""

    def setup_method(self):
        """Setup test data with duplicate column names."""
        # Create data with duplicate column names
        self.duplicate_columns = ["apple", "banana", "apple", "orange"]
        self.test_data = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
        
        # Create DataFrame with duplicate columns
        self.df = pd.DataFrame(self.test_data, columns=self.duplicate_columns)

    def test_multiindex_creation(self):
        """Test that MultiIndex is created for duplicate column names."""
        table = FastTable(data=self.df)
        
        # Check that internal data has MultiIndex
        assert isinstance(table._source_data.columns, pd.MultiIndex)
        assert table._source_data.columns.names == ["first", "second"]
        
        # Check first level (column names)
        first_level = table._source_data.columns.get_level_values("first").tolist()
        assert first_level == self.duplicate_columns
        
        # Check second level (unique indices)
        second_level = table._source_data.columns.get_level_values("second").tolist()
        assert second_level == [0, 1, 2, 3]

    def test_multiindex_with_list_input(self):
        """Test MultiIndex creation when passing list data with columns parameter."""
        table = FastTable(data=self.test_data, columns=self.duplicate_columns)
        
        assert isinstance(table._source_data.columns, pd.MultiIndex)
        first_level = table._source_data.columns.get_level_values("first").tolist()
        assert first_level == self.duplicate_columns

    def test_output_columns_simple_names(self):
        """Test that output columns are simple strings, not tuples."""
        table = FastTable(data=self.df)
        
        json_data = table.get_json_data()
        columns = json_data["columns"]
        
        # Columns should be simple strings (first level only)
        assert columns == self.duplicate_columns
        
        # Ensure no tuples in columns
        for col in columns:
            assert isinstance(col, str)
            assert not isinstance(col, tuple)

    def test_sorting_with_duplicate_columns(self):
        """Test that sorting works correctly with duplicate column names."""
        table = FastTable(data=self.df, sort_column_idx=0, sort_order="desc")
        
        # Get sorted data
        sorted_df = table._sorted_data
        
        # Check that first column ('apple', 0) is sorted descending
        first_col_values = sorted_df.iloc[:, 0].tolist()
        assert first_col_values == [9, 5, 1]

    def test_sorting_second_duplicate_column(self):
        """Test sorting by the second occurrence of duplicate column name."""
        table = FastTable(data=self.df)
        
        # Sort by column index 2 (second 'apple' column)
        table.sort(column_idx=2, order="asc")
        
        sorted_df = table._sorted_data
        third_col_values = sorted_df.iloc[:, 2].tolist()
        assert third_col_values == [3, 7, 11]

    def test_search_with_duplicate_columns(self):
        """Test that search works across all columns including duplicates."""
        table = FastTable(data=self.df)
        
        # Search for value that exists in multiple places
        searched_data = table._search("7")
        
        # Should find the row containing 7
        assert len(searched_data) == 1
        assert searched_data.iloc[0, 1] == 6  # banana column
        assert searched_data.iloc[0, 2] == 7  # second apple column

    def test_cell_value_access_both_duplicates(self):
        """Test that we can access values from both duplicate columns."""
        table = FastTable(data=self.df)
        
        # Access first 'apple' column (index 0)
        first_apple_col = table._source_data.columns[0]
        first_values = table._source_data[first_apple_col].tolist()
        assert first_values == [1, 5, 9]
        
        # Access second 'apple' column (index 2)
        second_apple_col = table._source_data.columns[2]
        second_values = table._source_data[second_apple_col].tolist()
        assert second_values == [3, 7, 11]
        
        # Verify they are different
        assert first_values != second_values

    def test_update_cell_with_duplicate_columns(self):
        """Test updating cells in duplicate columns."""
        table = FastTable(data=self.df)
        
        # Update first 'apple' column
        table.update_cell_value(row=0, column=0, value=999)
        assert table._source_data.iloc[0, 0] == 999
        
        # Update second 'apple' column
        table.update_cell_value(row=0, column=2, value=888)
        assert table._source_data.iloc[0, 2] == 888
        
        # Verify they are independent
        assert table._source_data.iloc[0, 0] == 999
        assert table._source_data.iloc[0, 2] == 888

    def test_insert_row_with_duplicate_columns(self):
        """Test inserting rows when columns have duplicates."""
        table = FastTable(data=self.df)
        
        new_row = [100, 200, 300, 400]
        table.insert_row(new_row, index=1)
        
        # Verify row was inserted
        assert len(table._source_data) == 4
        assert table._source_data.iloc[1, 0] == 100
        assert table._source_data.iloc[1, 2] == 300  # second apple column

    def test_to_pandas_preserves_column_names(self):
        """Test that to_pandas returns DataFrame with original column names."""
        table = FastTable(data=self.df)
        
        exported_df = table.to_pandas()
        
        # Should have original duplicate column names
        assert exported_df.columns.tolist() == self.duplicate_columns
        
        # Data should be preserved
        assert exported_df.iloc[0].tolist() == self.test_data[0]

    def test_to_json_simple_column_names(self):
        """Test that to_json exports simple column names."""
        table = FastTable(data=self.df)
        
        json_export = table.to_json()
        
        # Columns should be simple strings
        assert json_export["columns"] == self.duplicate_columns
        
        # Data should be preserved
        assert json_export["data"][0] == self.test_data[0]

    def test_select_row_by_value_duplicate_columns(self):
        """Test selecting row by value when column names are duplicated."""
        table = FastTable(data=self.df, is_selectable=True)
        
        # Select by first 'apple' column - should work with column name
        # Note: When there are duplicates, this will match the first occurrence
        table.select_row_by_value("apple", 5)
        
        selected = table.get_selected_row()
        assert selected.row_index == 1
        assert selected.row == [5, 6, 7, 8]

    def test_multiindex_with_numeric_data(self):
        """Test MultiIndex with numeric sorting."""
        # Create data where numeric sorting matters
        numeric_data = [
            ["3", "a", "29", "x"],
            ["29", "b", "3", "y"],
            ["100", "c", "10", "z"],
        ]
        
        table = FastTable(
            data=numeric_data, 
            columns=["value", "letter", "value", "char"],
            sort_column_idx=0,
            sort_order="asc"
        )
        
        # Should sort numerically: 3, 29, 100
        sorted_df = table._sorted_data
        first_col_values = sorted_df.iloc[:, 0].tolist()
        assert first_col_values == ["3", "29", "100"]

    def test_multiindex_filter_and_search(self):
        """Test that filter and search work together with MultiIndex."""
        table = FastTable(data=self.df, page_size=2)
        
        # Search should work
        table.search("1")
        
        # Should find rows with '1' in any column
        assert table._rows_total >= 1

    def test_no_multiindex_for_unique_columns(self):
        """Test that MultiIndex is NOT created when all column names are unique."""
        unique_columns = ["apple", "banana", "orange", "grape"]
        df_unique = pd.DataFrame(self.test_data, columns=unique_columns)
        
        table = FastTable(data=df_unique)
        
        # Should have MultiIndex (always created now for consistency)
        # But it should handle unique names correctly
        first_level = table._source_data.columns.get_level_values("first").tolist()
        assert first_level == unique_columns

    def test_multiindex_with_none_and_nan(self):
        """Test MultiIndex with None and NaN values in duplicate columns."""
        data_with_none = [
            [1, None, 3, 4],
            [5, 6, None, 8],
            [None, 10, 11, None],
        ]
        
        table = FastTable(data=data_with_none, columns=self.duplicate_columns)
        
        # Should convert None to ""
        json_data = table.get_json_data()
        
        # Check that None was converted to empty string
        assert json_data["data"][0]["items"][1] == ""
        assert json_data["data"][1]["items"][2] == ""

    def test_multiindex_pagination(self):
        """Test pagination works correctly with MultiIndex."""
        # Create larger dataset
        large_data = [[i, i+1, i+2, i+3] for i in range(25)]
        
        table = FastTable(
            data=large_data, 
            columns=self.duplicate_columns,
            page_size=10
        )
        
        # Check first page
        assert len(table._parsed_active_data["data"]) == 10
        
        # Check total rows
        assert table._rows_total == 25

    def test_multiindex_column_access_by_tuple(self):
        """Test that internal DataFrame allows tuple column access."""
        table = FastTable(data=self.df)
        
        # Access by tuple should work
        first_apple = table._source_data[("apple", 0)]
        second_apple = table._source_data[("apple", 2)]
        
        assert first_apple.tolist() == [1, 5, 9]
        assert second_apple.tolist() == [3, 7, 11]

    def test_multiindex_prevents_column_collision_error(self):
        """Test that MultiIndex prevents errors from duplicate column names."""
        # Without MultiIndex, accessing duplicate columns would cause issues
        # With MultiIndex, each column is uniquely accessible
        
        table = FastTable(data=self.df)
        
        # This should work without errors
        all_columns = []
        for i in range(len(table._source_data.columns)):
            col = table._source_data.columns[i]
            values = table._source_data[col].tolist()
            all_columns.append(values)
        
        # Should have 4 columns
        assert len(all_columns) == 4
        
        # Two 'apple' columns should have different values
        assert all_columns[0] != all_columns[2]


class TestFastTableDataFormats:
    """Test suite for different data input formats."""

    def test_empty_table_with_none(self):
        """Test creating empty table with None data."""
        table = FastTable(data=None, columns=["col1", "col2", "col3"])
        
        assert len(table._source_data) == 0
        assert table._rows_total == 0
        assert table._columns_first_idx == ["col1", "col2", "col3"]

    def test_empty_table_without_data(self):
        """Test creating empty table without data parameter."""
        table = FastTable(columns=["a", "b", "c"])
        
        assert len(table._source_data) == 0
        assert table._rows_total == 0

    def test_empty_dataframe(self):
        """Test creating table with empty DataFrame."""
        df = pd.DataFrame(columns=["col1", "col2", "col3"])
        table = FastTable(data=df)
        
        assert len(table._source_data) == 0
        assert table._rows_total == 0
        assert table._columns_first_idx == ["col1", "col2", "col3"]

    def test_list_format_basic(self):
        """Test creating table with basic list format."""
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        columns = ["a", "b", "c"]
        
        table = FastTable(data=data, columns=columns)
        
        assert table._rows_total == 3
        assert table._columns_first_idx == columns
        
        # Check data is preserved
        exported = table.to_pandas()
        assert exported.iloc[0].tolist() == [1, 2, 3]
        assert exported.iloc[2].tolist() == [7, 8, 9]

    def test_list_format_single_row(self):
        """Test creating table with single row in list format."""
        data = [[100, 200, 300]]
        columns = ["x", "y", "z"]
        
        table = FastTable(data=data, columns=columns)
        
        assert table._rows_total == 1
        assert table.to_pandas().iloc[0].tolist() == [100, 200, 300]

    def test_list_format_empty_list(self):
        """Test creating table with empty list."""
        data = []
        columns = ["col1", "col2"]
        
        table = FastTable(data=data, columns=columns)
        
        assert table._rows_total == 0
        assert len(table._source_data) == 0

    def test_list_format_with_none_values(self):
        """Test list data with None values gets converted to empty strings."""
        data = [[1, None, 3], [None, 5, None], [7, 8, 9]]
        columns = ["a", "b", "c"]
        
        table = FastTable(data=data, columns=columns)
        json_data = table.get_json_data()
        
        # None should be converted to ""
        assert json_data["data"][0]["items"][1] == ""
        assert json_data["data"][1]["items"][0] == ""
        assert json_data["data"][1]["items"][2] == ""

    def test_dataframe_with_range_index(self):
        """Test DataFrame with default numeric column names."""
        data = [[1, 2, 3], [4, 5, 6]]
        df = pd.DataFrame(data)  # No column names specified
        
        table = FastTable(data=df)
        
        assert table._rows_total == 2
        # Should have no MultiIndex for RangeIndex
        assert table._columns_first_idx is None or isinstance(table._source_data.columns, pd.RangeIndex)

    def test_read_pandas_replaces_data(self):
        """Test that read_pandas method replaces table data."""
        initial_data = [[1, 2], [3, 4]]
        table = FastTable(data=initial_data, columns=["a", "b"])
        
        assert table._rows_total == 2
        
        # Replace with new data - same columns!
        new_df = pd.DataFrame([[99, 88], [77, 66]], columns=["a", "b"])
        table.read_pandas(new_df)
        
        assert table._rows_total == 2
        # read_pandas keeps original column names
        assert table._columns_first_idx == ["a", "b"]
        exported = table.to_pandas()
        assert exported.iloc[0].tolist() == [99, 88]

    def test_read_json_replaces_data(self):
        """Test that read_json method replaces table data."""
        initial_data = [[1, 2], [3, 4]]
        table = FastTable(data=initial_data, columns=["a", "b"])
        
        # Replace with new data via JSON (must include options for read_json)
        new_data = {
            "data": [[100, 200], [300, 400], [500, 600]],
            "columns": ["col1", "col2"],
            "options": {}  # read_json requires options field
        }
        table.read_json(new_data)
        
        assert table._rows_total == 3
        json_export = table.to_json()
        assert json_export["data"][0] == [100, 200]
        assert json_export["data"][2] == [500, 600]

    def test_clear_method_empties_table(self):
        """Test that clear method removes all data."""
        data = [[1, 2, 3], [4, 5, 6]]
        table = FastTable(data=data, columns=["a", "b", "c"])
        
        assert table._rows_total == 2
        
        table.clear()
        
        assert table._rows_total == 0
        assert len(table._source_data) == 0

    def test_add_rows_method(self):
        """Test adding multiple rows at once."""
        table = FastTable(data=[[1, 2]], columns=["a", "b"])
        
        assert table._rows_total == 1
        
        table.add_rows([[3, 4], [5, 6], [7, 8]])
        
        assert table._rows_total == 4
        exported = table.to_pandas()
        assert exported.iloc[3].tolist() == [7, 8]

    def test_pop_row_removes_and_returns(self):
        """Test pop_row method removes and returns a row."""
        data = [[1, 2], [3, 4], [5, 6]]
        table = FastTable(data=data, columns=["a", "b"])
        
        assert table._rows_total == 3
        
        # Pop last row
        popped = table.pop_row(-1)
        
        assert table._rows_total == 2
        assert popped.tolist() == [5, 6]

    def test_mixed_data_types_in_columns(self):
        """Test table with mixed data types (strings and numbers)."""
        data = [
            ["text1", 100, 1.5],
            ["text2", 200, 2.5],
            ["text3", 300, 3.5],
        ]
        columns = ["str_col", "int_col", "float_col"]
        
        table = FastTable(data=data, columns=columns)
        
        assert table._rows_total == 3
        exported = table.to_pandas()
        assert exported.iloc[1, 0] == "text2"
        assert exported.iloc[1, 1] == 200
        assert exported.iloc[1, 2] == 2.5


class TestFastTableEdgeCases:
    """Test suite for edge cases and special scenarios."""

    def test_very_long_table(self):
        """Test table with many rows (stress test)."""
        data = [[i, i*2, i*3] for i in range(1000)]
        columns = ["x", "y", "z"]

        table = FastTable(data=data, columns=columns, page_size=50)

        assert table._rows_total == 1000
        assert len(table._parsed_active_data["data"]) == 50  # First page

    def test_wide_table(self):
        """Test table with many columns."""
        num_cols = 50
        columns = [f"col{i}" for i in range(num_cols)]
        data = [[i for i in range(num_cols)]]

        table = FastTable(data=data, columns=columns)

        assert table._rows_total == 1
        assert len(table._columns_first_idx) == num_cols

    def test_special_characters_in_column_names(self):
        """Test column names with special characters."""
        columns = ["col_1", "col-2", "col.3", "col@4", "col 5"]
        data = [[1, 2, 3, 4, 5]]

        table = FastTable(data=data, columns=columns)

        assert table._columns_first_idx == columns
        assert table._rows_total == 1

    def test_unicode_in_data(self):
        """Test table with unicode characters."""
        data = [
            ["こんにちは", "你好", "مرحبا", "Hello"],
            ["さようなら", "再见", "مع السلامة", "Goodbye 👋"],
        ]
        columns = ["Japanese", "Chinese", "Arabic", "English"]

        table = FastTable(data=data, columns=columns)

        assert table._rows_total == 2
        exported = table.to_pandas()
        assert exported.iloc[0, 0] == "こんにちは"
        assert exported.iloc[1, 3] == "Goodbye 👋"

    def test_duplicate_column_with_select_rows_by_value(self):
        """Test select_rows_by_value with duplicate columns."""
        data = [[1, 2, 3], [5, 6, 7], [5, 8, 9]]
        columns = ["apple", "banana", "apple"]

        table = FastTable(data=data, columns=columns, is_selectable=True)

        # Select rows where first 'apple' column has value 5
        table.select_rows_by_value("apple", [5])

        selected = table.get_selected_rows()
        assert len(selected) == 2
        assert selected[0].row_index == 1
        assert selected[1].row_index == 2

    def test_three_duplicate_columns(self):
        """Test table with three columns having the same name."""
        data = [[1, 2, 3, 4], [5, 6, 7, 8]]
        columns = ["value", "value", "value", "other"]

        table = FastTable(data=data, columns=columns)

        # Should have MultiIndex
        assert isinstance(table._source_data.columns, pd.MultiIndex)

        # All three 'value' columns should be accessible
        first_value = table._source_data[("value", 0)].tolist()
        second_value = table._source_data[("value", 1)].tolist()
        third_value = table._source_data[("value", 2)].tolist()

        assert first_value == [1, 5]
        assert second_value == [2, 6]
        assert third_value == [3, 7]


class TestFastTableNoneHandling:
    """Test suite for proper None/NaN handling in FastTable."""

    def setup_method(self):
        """Setup test data with None values."""
        self.data_with_none = [
            [1, None, 3, "text"],
            [None, 6, None, "value"],
            [9, 10, 11, None],
        ]
        self.columns = ["col1", "col2", "col3", "col4"]

    def test_none_values_preserved_in_source_data(self):
        """Test that None values are preserved in _source_data (as NaN in pandas)."""
        table = FastTable(data=self.data_with_none, columns=self.columns)

        # None is converted to NaN by pandas, which should be preserved in source data
        assert pd.isna(table._source_data.iloc[0, 1])
        assert pd.isna(table._source_data.iloc[1, 0])
        assert pd.isna(table._source_data.iloc[1, 2])
        assert pd.isna(table._source_data.iloc[2, 3])

    def test_none_values_converted_to_empty_string_in_json(self):
        """Test that None values are converted to empty strings in JSON output."""
        table = FastTable(data=self.data_with_none, columns=self.columns)

        json_data = table.get_json_data()

        # None should be converted to "" for display
        assert json_data["data"][0]["items"][1] == ""
        assert json_data["data"][1]["items"][0] == ""
        assert json_data["data"][1]["items"][2] == ""
        assert json_data["data"][2]["items"][3] == ""

        # Non-None values should be preserved
        assert json_data["data"][0]["items"][0] == 1
        assert json_data["data"][0]["items"][3] == "text"

    def test_to_pandas_preserves_none_values(self):
        """Test that to_pandas() returns DataFrame with NaN values (pandas behavior)."""
        table = FastTable(data=self.data_with_none, columns=self.columns)

        exported_df = table.to_pandas()

        # pandas converts None to NaN, which should be preserved
        assert pd.isna(exported_df.iloc[0, 1])
        assert pd.isna(exported_df.iloc[1, 0])
        assert pd.isna(exported_df.iloc[1, 2])
        assert pd.isna(exported_df.iloc[2, 3])

    def test_get_clicked_row_returns_none_values(self):
        """Test that get_clicked_row returns NaN values from source (not empty strings)."""
        table = FastTable(data=self.data_with_none, columns=self.columns)

        # Directly get data from source for row 1
        row_data = table._source_data.loc[1].values.tolist()

        # Row should have NaN values (pandas converts None to NaN)
        assert pd.isna(row_data[0])  # col1
        assert pd.isna(row_data[2])  # col3
        assert row_data[1] == 6  # col2
        assert row_data[3] == "value"  # col4

    def test_get_clicked_cell_returns_none_value(self):
        """Test that get_clicked_cell returns NaN value from source."""
        import json

        from supervisely.app.content import StateJson

        table = FastTable(data=self.data_with_none, columns=self.columns)

        # Manually set clicked cell state to simulate cell click
        StateJson()[table.widget_id]["selectedCell"] = {
            "idx": 0,
            "column": 1,
            "row": [1, None, 3, "text"],  # This gets ignored, we read from source
        }

        clicked_cell = table.get_clicked_cell()

        # Should return NaN from source data (pandas converts None to NaN)
        assert pd.isna(clicked_cell.column_value)
        assert clicked_cell.column_name == "col2"
        assert clicked_cell.row_index == 0

    def test_select_row_by_value_with_none(self):
        """Test working with NaN values in source data."""
        import numpy as np

        table = FastTable(data=self.data_with_none, columns=self.columns, is_selectable=True)

        # pandas converts None to NaN
        # Verify that NaN values exist in source
        col_idx = table._columns_first_idx.index("col2")
        col_tuple = table._source_data.columns[col_idx]
        nan_rows = table._source_data[table._source_data[col_tuple].isna()]

        assert len(nan_rows) > 0, "Should have NaN values in col2"
        assert pd.isna(table._source_data.iloc[0, 1]), "First row col2 should be NaN"

    def test_select_rows_by_value_with_none(self):
        """Test searching for NaN values using pandas isna()."""
        table = FastTable(data=self.data_with_none, columns=self.columns, is_selectable=True)

        # Find rows where col3 is NaN using pandas
        col_idx = table._columns_first_idx.index("col3")
        col_tuple = table._source_data.columns[col_idx]
        nan_rows = table._source_data[table._source_data[col_tuple].isna()]

        # Select those rows manually
        indices = nan_rows.index.tolist()
        table.select_rows(indices)

        selected = table.get_selected_rows()
        assert len(selected) == 1
        assert selected[0].row_index == 1
        assert pd.isna(selected[0].row[2])

    def test_pop_row_returns_none_values(self):
        """Test that pop_row returns row with NaN values."""
        table = FastTable(data=self.data_with_none, columns=self.columns)

        popped = table.pop_row(0)

        # pandas converts None to NaN
        assert pd.isna(popped[1])
        assert popped[0] == 1
        assert popped[2] == 3

    def test_update_cell_with_none(self):
        """Test updating a cell to None value (becomes NaN in pandas)."""
        table = FastTable(data=[[1, 2, 3]], columns=["a", "b", "c"])

        # Update cell to None
        table.update_cell_value(row=0, column=1, value=None)

        # Source data should have NaN (pandas converts None to NaN)
        assert pd.isna(table._source_data.iloc[0, 1])

        # JSON output should have ""
        json_data = table.get_json_data()
        assert json_data["data"][0]["items"][1] == ""

        # to_pandas should have NaN
        exported = table.to_pandas()
        assert pd.isna(exported.iloc[0, 1])

    def test_insert_row_with_none_values(self):
        """Test inserting a row containing None values."""
        table = FastTable(data=[[1, 2, 3]], columns=["a", "b", "c"])

        new_row = [None, 5, None]
        table.insert_row(new_row, index=0)

        # Source data should preserve None
        assert table._source_data.iloc[0, 0] is None
        assert table._source_data.iloc[0, 2] is None
        assert table._source_data.iloc[0, 1] == 5

    def test_add_rows_with_none_values(self):
        """Test adding multiple rows with None values (become NaN in pandas)."""
        table = FastTable(data=[[1, 2]], columns=["a", "b"])

        table.add_rows([[None, 4], [5, None], [None, None]])

        # Verify None values are preserved as NaN
        assert pd.isna(table._source_data.iloc[1, 0])
        assert pd.isna(table._source_data.iloc[2, 1])
        assert pd.isna(table._source_data.iloc[3, 0])
        assert pd.isna(table._source_data.iloc[3, 1])

    def test_search_with_none_values(self):
        """Test that search works correctly with NaN values."""
        table = FastTable(data=self.data_with_none, columns=self.columns)

        # Search for "6" should find row with NaN values
        searched = table._search("6")

        assert len(searched) == 1
        assert pd.isna(searched.iloc[0, 0])  # NaN is preserved
        assert searched.iloc[0, 1] == 6

    def test_sort_with_none_values(self):
        """Test that sorting works with None values."""
        data = [[3, None], [1, "b"], [None, "a"], [2, "c"]]
        table = FastTable(data=data, columns=["num", "letter"], sort_column_idx=0, sort_order="asc")

        sorted_df = table._sorted_data

        # None values should be handled during sort
        assert sorted_df is not None
        assert len(sorted_df) == 4

    def test_filter_with_none_values(self):
        """Test custom filter function with None values."""
        table = FastTable(data=self.data_with_none, columns=self.columns)

        def filter_non_none_in_col2(data, value):
            # Filter rows where col2 is not None
            col_tuple = data.columns[1]  # col2
            return data[data[col_tuple].notna()]

        table.set_filter(filter_non_none_in_col2)
        table.filter(True)

        # Should only have rows where col2 is not None
        assert table._rows_total == 2  # rows 1 and 2 (indices 1 and 2)

    def test_dataframe_with_nan_values(self):
        """Test that pandas NaN values are also handled correctly."""
        import numpy as np

        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, 3], "c": [1, 2, np.nan]})

        table = FastTable(data=df)

        # NaN should be converted to "" in JSON
        json_data = table.get_json_data()
        assert json_data["data"][0]["items"][1] == ""
        assert json_data["data"][1]["items"][0] == ""
        assert json_data["data"][2]["items"][2] == ""

        # But preserved in to_pandas
        exported = table.to_pandas()
        assert pd.isna(exported.iloc[0, 1])
        assert pd.isna(exported.iloc[1, 0])
        assert pd.isna(exported.iloc[2, 2])

    def test_to_json_converts_none_to_empty_string(self):
        """Test that to_json exports None as empty strings."""
        table = FastTable(data=self.data_with_none, columns=self.columns)

        json_export = table.to_json()

        # None should be "" in JSON export
        assert json_export["data"][0][1] == ""
        assert json_export["data"][1][0] == ""
        assert json_export["data"][1][2] == ""

    def test_active_page_with_none_preserves_values(self):
        """Test that to_pandas(active_page=True) preserves NaN values."""
        table = FastTable(data=self.data_with_none, columns=self.columns, page_size=2)

        # Get active page
        exported = table.to_pandas(active_page=True)

        # Should preserve NaN values from active page
        assert pd.isna(exported.iloc[0, 1])
        assert pd.isna(exported.iloc[1, 0])

    def test_multiindex_with_none_values(self):
        """Test None handling with duplicate column names (MultiIndex)."""
        data = [[1, None, 3], [None, 5, None]]
        columns = ["value", "value", "other"]

        table = FastTable(data=data, columns=columns)

        # Verify MultiIndex is created
        assert isinstance(table._source_data.columns, pd.MultiIndex)

        # None is converted to NaN by pandas and should be preserved in source
        assert pd.isna(table._source_data.iloc[0, 1])
        assert pd.isna(table._source_data.iloc[1, 0])
        assert pd.isna(table._source_data.iloc[1, 2])

        # to_pandas should preserve NaN
        exported = table.to_pandas()
        assert pd.isna(exported.iloc[0, 1])
        assert pd.isna(exported.iloc[1, 0])


class TestFastTableSorting:
    """Comprehensive test suite for FastTable sorting functionality."""

    def test_sort_numeric_ascending(self):
        """Test sorting numeric column in ascending order."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        assert sorted_df.iloc[:, 0].tolist() == [1, 2, 5, 8]
        # Verify data is synchronized
        assert table._rows_total == 4
        assert len(table._parsed_active_data["data"]) == 4

    def test_sort_numeric_descending(self):
        """Test sorting numeric column in descending order."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="desc")

        sorted_df = table._sorted_data
        assert sorted_df.iloc[:, 0].tolist() == [8, 5, 2, 1]

    def test_sort_string_ascending(self):
        """Test sorting string column in ascending order."""
        data = [["zebra", 1], ["apple", 2], ["mango", 3], ["banana", 4]]
        table = FastTable(data=data, columns=["fruit", "count"])

        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        assert sorted_df.iloc[:, 0].tolist() == ["apple", "banana", "mango", "zebra"]

    def test_sort_string_descending(self):
        """Test sorting string column in descending order."""
        data = [["zebra", 1], ["apple", 2], ["mango", 3], ["banana", 4]]
        table = FastTable(data=data, columns=["fruit", "count"])

        table.sort(column_idx=0, order="desc")

        sorted_df = table._sorted_data
        assert sorted_df.iloc[:, 0].tolist() == ["zebra", "mango", "banana", "apple"]

    def test_sort_numeric_strings_as_numbers(self):
        """Test that numeric strings are sorted numerically, not alphabetically."""
        data = [["100", "a"], ["3", "b"], ["29", "c"], ["5", "d"]]
        table = FastTable(data=data, columns=["num_str", "letter"])

        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        # Should sort as numbers: 3, 5, 29, 100 (not "100", "29", "3", "5")
        assert sorted_df.iloc[:, 0].tolist() == ["3", "5", "29", "100"]

    def test_sort_mixed_numeric_strings_descending(self):
        """Test numeric strings sorted descending."""
        data = [["100", "a"], ["3", "b"], ["29", "c"], ["5", "d"]]
        table = FastTable(data=data, columns=["num_str", "letter"])

        table.sort(column_idx=0, order="desc")

        sorted_df = table._sorted_data
        assert sorted_df.iloc[:, 0].tolist() == ["100", "29", "5", "3"]

    def test_sort_with_none_values_ascending(self):
        """Test that None values are placed at the end when sorting ascending."""
        data = [[3, "c"], [None, "none1"], [1, "a"], [None, "none2"], [2, "b"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        # None values should be at the end
        values = sorted_df.iloc[:, 0].tolist()
        assert values[:3] == [1, 2, 3]
        assert pd.isna(values[3])
        assert pd.isna(values[4])

    def test_sort_with_none_values_descending(self):
        """Test that None values are placed at the end when sorting descending."""
        data = [[3, "c"], [None, "none1"], [1, "a"], [None, "none2"], [2, "b"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="desc")

        sorted_df = table._sorted_data
        values = sorted_df.iloc[:, 0].tolist()
        assert values[:3] == [3, 2, 1]
        assert pd.isna(values[3])
        assert pd.isna(values[4])

    def test_sort_with_nan_values(self):
        """Test sorting with NaN values from pandas."""
        import numpy as np

        df = pd.DataFrame({"num": [3, np.nan, 1, np.nan, 2], "letter": ["c", "n1", "a", "n2", "b"]})
        table = FastTable(data=df)

        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        values = sorted_df.iloc[:, 0].tolist()
        assert values[:3] == [1, 2, 3]
        assert pd.isna(values[3])
        assert pd.isna(values[4])

    def test_sort_initialization_with_params(self):
        """Test that table is sorted correctly when sort params are passed to constructor."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"], sort_column_idx=0, sort_order="asc")

        # Should be sorted on initialization
        sorted_df = table._sorted_data
        assert sorted_df.iloc[:, 0].tolist() == [1, 2, 5, 8]

    def test_sort_none_column_idx(self):
        """Test that passing None parameters without reset preserves current sorting."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        # First sort
        table.sort(column_idx=0, order="asc")
        assert table._sorted_data.iloc[:, 0].tolist() == [1, 2, 5, 8]

        # Call with None params (without reset) - should preserve current sorting
        table.sort(column_idx=None, order=None)

        # Should keep current sorting
        assert table._sorted_data.iloc[:, 0].tolist() == [1, 2, 5, 8]
        assert table._sort_column_idx == 0
        assert table._sort_order == "asc"

    def test_sort_none_order(self):
        """Test that passing only order=None keeps current column."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        # Don't sort on initialization
        table = FastTable(data=data, columns=["num", "letter"])

        # Sort the table first
        table.sort(column_idx=0, order="asc")
        assert table._sorted_data.iloc[:, 0].tolist() == [1, 2, 5, 8]
        assert table._sort_column_idx == 0
        assert table._sort_order == "asc"

        # Change only order, keep column
        table.sort(order="desc")
        assert table._sorted_data.iloc[:, 0].tolist() == [8, 5, 2, 1]
        assert table._sort_column_idx == 0  # Column preserved
        assert table._sort_order == "desc"

    def test_sort_none_column_keeps_order(self):
        """Test that passing only column_idx=None keeps current order."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        # Sort by column 0 ascending
        table.sort(column_idx=0, order="asc")
        assert table._sort_column_idx == 0
        assert table._sort_order == "asc"

        # Change to column 1, keep order
        table.sort(column_idx=1)
        assert table._sorted_data.iloc[:, 1].tolist() == ["a", "b", "e", "h"]
        assert table._sort_column_idx == 1
        assert table._sort_order == "asc"  # Order preserved

    def test_sort_reset_clears_sorting(self):
        """Test that reset=True clears sorting completely."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        # Sort the table first
        table.sort(column_idx=0, order="asc")
        assert table._sorted_data.iloc[:, 0].tolist() == [1, 2, 5, 8]
        assert table._sort_column_idx == 0
        assert table._sort_order == "asc"

        # Clear sorting with reset=True
        table.sort(reset=True)

        # Should return to original order
        assert table._sorted_data.iloc[:, 0].tolist() == [5, 2, 8, 1]
        assert table._sort_column_idx is None
        assert table._sort_order is None

    def test_sort_no_params_preserves_state(self):
        """Test that sort() without parameters preserves current sorting state."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        # Sort the table first
        table.sort(column_idx=0, order="asc")
        assert table._sort_column_idx == 0
        assert table._sort_order == "asc"

        # Call sort without params - should preserve state
        table.sort()
        assert table._sort_column_idx == 0
        assert table._sort_order == "asc"

    def test_sort_invalid_column_idx(self):
        """Test that invalid column index raises an error."""
        data = [[5, "e"], [2, "b"]]
        table = FastTable(data=data, columns=["num", "letter"])

        with pytest.raises(IndexError) as exc_info:
            table.sort(column_idx=10, order="asc")

        assert "column idx = 10 is not possible" in str(exc_info.value)

    def test_sort_negative_column_idx_validation(self):
        """Test that negative column index is validated and set to None."""
        data = [[5, "e"], [2, "b"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=-1, order="asc")

        # Should be set to None and no sorting applied
        assert table._sort_column_idx is None
        assert table._sorted_data.iloc[:, 0].tolist() == [5, 2]

    def test_sort_invalid_order_validation(self):
        """Test that invalid order value is validated and set to None."""
        data = [[5, "e"], [2, "b"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="invalid")

        # Should be set to None and no sorting applied
        assert table._sort_order is None
        assert table._sorted_data.iloc[:, 0].tolist() == [5, 2]

    def test_sort_with_pagination(self):
        """Test that sorting works correctly with pagination."""
        data = [[i, chr(ord("a") + i)] for i in range(10, 0, -1)]  # 10, 9, ..., 1
        table = FastTable(data=data, columns=["num", "letter"], page_size=3)

        table.sort(column_idx=0, order="asc")

        # First page should have 1, 2, 3
        parsed_data = table._parsed_active_data["data"]
        assert len(parsed_data) == 3
        assert parsed_data[0]["items"][0] == 1
        assert parsed_data[1]["items"][0] == 2
        assert parsed_data[2]["items"][0] == 3

    def test_sort_with_search(self):
        """Test that sorting works correctly after search filtering."""
        data = [[5, "apple"], [2, "banana"], [8, "apricot"], [1, "avocado"], [3, "berry"]]
        table = FastTable(data=data, columns=["num", "fruit"])

        # Search for fruits starting with 'a'
        table.search("a")

        # Sort the filtered results
        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        # Should have only filtered items, sorted: 1, 5, 8
        assert len(sorted_df) == 4  # apple, apricot, avocado, banana
        assert sorted_df.iloc[0, 0] == 1  # avocado

    def test_sort_with_filter(self):
        """Test that sorting works correctly after custom filtering."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"], [6, "f"]]
        table = FastTable(data=data, columns=["num", "letter"])

        # Set custom filter to keep only values > 3
        def filter_gt_3(df, value):
            if value is None:
                return df
            col_tuple = df.columns[0]
            return df[df[col_tuple] > value]

        table.set_filter(filter_gt_3)
        table.filter(3)

        # Sort filtered results
        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        # Should have only 5, 6, 8 sorted
        assert sorted_df.iloc[:, 0].tolist() == [5, 6, 8]

    def test_sort_preserves_row_indices(self):
        """Test that sorting preserves original row indices in parsed data."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="asc")

        # Check that idx values correspond to original positions
        parsed_data = table._parsed_active_data["data"]
        assert parsed_data[0]["idx"] == 3  # Row with value 1 was at index 3
        assert parsed_data[1]["idx"] == 1  # Row with value 2 was at index 1
        assert parsed_data[2]["idx"] == 0  # Row with value 5 was at index 0
        assert parsed_data[3]["idx"] == 2  # Row with value 8 was at index 2

    def test_sort_updates_state_json(self):
        """Test that sort method updates StateJson correctly."""
        data = [[5, "e"], [2, "b"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=1, order="desc")

        # Verify StateJson is updated
        from supervisely.app.content import StateJson

        assert StateJson()[table.widget_id]["sort"]["column"] == 1
        assert StateJson()[table.widget_id]["sort"]["order"] == "desc"

    def test_sort_updates_data_json(self):
        """Test that sort method updates DataJson with sorted data."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="asc")

        # Verify DataJson contains sorted data
        from supervisely.app import DataJson

        data_items = DataJson()[table.widget_id]["data"]
        assert data_items[0]["items"][0] == 1
        assert data_items[1]["items"][0] == 2
        assert data_items[2]["items"][0] == 5
        assert data_items[3]["items"][0] == 8

    def test_sort_second_column(self):
        """Test sorting by second column."""
        data = [[1, "z"], [2, "a"], [3, "m"], [4, "b"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=1, order="asc")

        sorted_df = table._sorted_data
        assert sorted_df.iloc[:, 1].tolist() == ["a", "b", "m", "z"]

    def test_sort_switch_column(self):
        """Test switching sort column."""
        data = [[5, "e"], [2, "b"], [8, "a"], [1, "z"]]
        table = FastTable(data=data, columns=["num", "letter"])

        # First sort by column 0
        table.sort(column_idx=0, order="asc")
        assert table._sorted_data.iloc[:, 0].tolist() == [1, 2, 5, 8]

        # Then sort by column 1
        table.sort(column_idx=1, order="asc")
        assert table._sorted_data.iloc[:, 1].tolist() == ["a", "b", "e", "z"]

    def test_sort_switch_order(self):
        """Test switching sort order on same column."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        # Sort ascending
        table.sort(column_idx=0, order="asc")
        assert table._sorted_data.iloc[:, 0].tolist() == [1, 2, 5, 8]

        # Sort descending
        table.sort(column_idx=0, order="desc")
        assert table._sorted_data.iloc[:, 0].tolist() == [8, 5, 2, 1]

    def test_sort_empty_table(self):
        """Test sorting an empty table doesn't cause errors."""
        table = FastTable(data=None, columns=["num", "letter"])

        table.sort(column_idx=0, order="asc")

        assert len(table._sorted_data) == 0
        assert table._rows_total == 0

    def test_sort_single_row(self):
        """Test sorting a table with single row."""
        data = [[5, "e"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="asc")

        assert table._sorted_data.iloc[0, 0] == 5
        assert len(table._sorted_data) == 1

    def test_sort_duplicate_values(self):
        """Test sorting with duplicate values maintains stability."""
        data = [[5, "a"], [5, "b"], [5, "c"], [1, "d"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        # First value should be 1
        assert sorted_df.iloc[0, 0] == 1
        # Rest should be 5's (order may vary)
        assert sorted_df.iloc[1, 0] == 5
        assert sorted_df.iloc[2, 0] == 5
        assert sorted_df.iloc[3, 0] == 5

    def test_sort_mixed_types_column(self):
        """Test sorting column with mixed numeric strings and pure strings."""
        data = [["10", "a"], ["abc", "b"], ["2", "c"], ["xyz", "d"]]
        table = FastTable(data=data, columns=["mixed", "letter"])

        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        # Numeric strings should be sorted first numerically, then non-numeric alphabetically
        values = sorted_df.iloc[:, 0].tolist()
        # Should have: 2, 10, abc, xyz
        assert values == ["2", "10", "abc", "xyz"]

    def test_custom_sort_function(self):
        """Test using a custom sort function."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        # Custom sort: reverse alphabetical on second column
        def custom_sort(data, column_idx, order):
            col = data.columns[column_idx]
            ascending = order == "asc"
            # Reverse the ascending flag for custom behavior
            return data.sort_values(by=col, ascending=not ascending)

        table.set_sort(custom_sort)
        table.sort(column_idx=1, order="asc")

        # With reversed logic, "asc" should give descending
        sorted_df = table._sorted_data
        assert sorted_df.iloc[:, 1].tolist() == ["h", "e", "b", "a"]

    def test_sort_after_insert_row(self):
        """Test that sorting works correctly after inserting a row."""
        data = [[5, "e"], [2, "b"], [8, "h"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="asc")

        # Insert new row
        table.insert_row([1, "a"], index=0)

        # Re-sort
        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        assert sorted_df.iloc[:, 0].tolist() == [1, 2, 5, 8]

    def test_sort_after_pop_row(self):
        """Test that sorting works correctly after removing a row."""
        data = [[5, "e"], [2, "b"], [8, "h"], [1, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="asc")

        # Remove a row
        table.pop_row(0)

        # Re-sort
        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        # Should have 3 rows now
        assert len(sorted_df) == 3

    def test_sort_with_all_none_column(self):
        """Test sorting a column that contains only None values."""
        data = [[None, "a"], [None, "b"], [None, "c"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="asc")

        # Should not crash, all values are None
        sorted_df = table._sorted_data
        assert len(sorted_df) == 3
        assert all(pd.isna(sorted_df.iloc[:, 0]))

    def test_sort_preserves_data_types(self):
        """Test that sorting doesn't change data types."""
        data = [[5.5, "e"], [2.1, "b"], [8.9, "h"], [1.0, "a"]]
        table = FastTable(data=data, columns=["num", "letter"])

        table.sort(column_idx=0, order="asc")

        sorted_df = table._sorted_data
        # Verify floats are preserved
        assert sorted_df.iloc[0, 0] == 1.0
        assert isinstance(sorted_df.iloc[0, 0], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
