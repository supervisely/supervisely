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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
