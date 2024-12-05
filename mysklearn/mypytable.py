"""
Programmer: Murat
Class: CPSC 322, Fall 2024
Programming Assignment: 3
Date: 11/7/2024
I attempted the bonus: Yes


"""

from mysklearn import myutils

import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """

        column = []

        # Identify the column index based on the identifier (name or index)
        if isinstance(col_identifier, str):
            try:
                column_id = self.column_names.index(col_identifier)
            except ValueError as exc:
                raise ValueError(f"Column name '{col_identifier}' not found in column_names.") from exc
        elif isinstance(col_identifier, int):
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError("Column index is out of range.")
            column_id = col_identifier
        else:
            raise TypeError("Invalid data type for col_identifier. Expected str or int.")

        for row in self.data:
            if include_missing_values:
                column.append(row[column_id])
            else:
                if row[column_id] != "NA":
                    column.append(row[column_id])

        return column

    def encode_categorical_columns(self):
        """Encodes categorical columns into numerical values."""
        encoders = {}

        for col_name in self.column_names:
            col_index = self.column_names.index(col_name)
            unique_values = set(str(row[col_index]) for row in self.data)

            value_to_num = {value: idx for idx, value in enumerate(sorted(unique_values))}
            encoders[col_name] = value_to_num

            for row in self.data:
                row[col_index] = value_to_num[str(row[col_index])]

        return encoders


    def update_column(self, column_name, new_values):
        col_index = self.column_names.index(column_name)
        for i, row in enumerate(self.data):
            row[col_index] = new_values[i]

    def drop_columns(self, columns_to_drop):
        """
        Removes specified columns from the table.

        Args:
            columns_to_drop (list of str): List of column names to remove.

        Returns:
            MyPyTable: A new MyPyTable instance with the specified columns removed.
        """
        drop_indices = [self.column_names.index(col) for col in columns_to_drop]

        new_column_names = [
            col for i, col in enumerate(self.column_names) if i not in drop_indices
        ]

        new_data = [
            [value for i, value in enumerate(row) if i not in drop_indices]
            for row in self.data
        ]

        return MyPyTable(column_names=new_column_names, data=new_data)


    def get_data_types(self):
        """Infers and returns the data type for each column in the table.

        Returns:
            dict: A dictionary where keys are column names and values are data types.
        """
        data_types = {}
        for col in self.column_names:
            col_data = self.get_column(col)
            inferred_type = None
            for value in col_data:
                if value != "NA":
                    if inferred_type is None:
                        inferred_type = type(value)
                    elif type(value) != inferred_type:
                        inferred_type = str
                        break
            data_types[col] = inferred_type.__name__ if inferred_type else "unknown"
        return data_types

    def normalize_columns(self, columns_to_normalize):
        """Normalize specified columns using Min-Max Scaling.

        Args:
            columns_to_normalize (list of str): List of column names to normalize.

        Notes:
            Values are scaled to be in the range [0, 1].
        """
        for col in columns_to_normalize:
            col_data = self.get_column(col, include_missing_values=False)
            if not col_data:
                continue  # Skip empty columns
            min_val = min(col_data)
            max_val = max(col_data)
            if max_val == min_val:
                continue  # Skip if all values are the same (avoid division by zero)
            for row in self.data:
                if row[self.column_names.index(col)] != "NA":
                    row[self.column_names.index(col)] = (
                        row[self.column_names.index(col)] - min_val
                    ) / (max_val - min_val)


    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i, row in enumerate(self.data):
            for j, value in enumerate(row):
                try:
                    self.data[i][j] = float(value)
                except ValueError:
                    continue

        return self

    def convert_column_to_type(self, column_name, target_type):
        """Convert the data type of a specified column to the given target type.

        Args:
            column_name (str): The name of the column to convert.
            target_type (type): The target data type (e.g., float, int, str).

        Notes:
            If a value cannot be converted, it is replaced with "NA".
        """
        if column_name not in self.column_names:
            raise ValueError(f"Column '{column_name}' does not exist in the table.")

        col_index = self.column_names.index(column_name)
        for row in self.data:
            try:
                row[col_index] = target_type(row[col_index])
            except (ValueError, TypeError):
                row[col_index] = "NA"
        return self


    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        for row in sorted(row_indexes_to_drop,reverse=True):
            if 0 <= row < len(self.data):
                self.data.pop(row)
        return self

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """

        with open(filename, newline='', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            self.data = list(csvreader)
            if len(self.data) == 0:
                raise ValueError(f"The file '{filename}' is empty or does not contain valid data.")
            self.column_names = self.data.pop(0)
            self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open( filename, mode='w', newline = '', encoding = 'utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            if self.column_names:
                csvwriter.writerow(self.column_names)
            csvwriter.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        key_col_id = []
        for key_column_name in key_column_names:
            key_col_id.append(self.column_names.index(key_column_name))

        answer = set()
        for i, _ in enumerate(self.data):
            values = []
            for key_id in key_col_id:
                values.append(self.data[i][key_id])
            for j in range(i + 1, len(self.data)):
                values_compare_to = []
                for key_id in key_col_id:
                    values_compare_to.append(self.data[j][key_id])
                if values == values_compare_to:
                    answer.add(j)
        return sorted(list(answer))



    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        rows_to_drop = []
        for i, row in enumerate(self.data):
            if "NA" in row:
                rows_to_drop.append(i)
        self.drop_rows(rows_to_drop)
        return self

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """

        col_id = self.column_names.index(col_name)
        column = []
        missing_rows = []
        avg = 0
        for i, row in enumerate(self.data):
            if row[col_id] != "NA":
                try:
                    column.append(float(row[col_id]))
                except ValueError as error:
                    raise ValueError(f"Cannot convert {row[col_id]} to float") from error
            else:
                missing_rows.append(i)
        if column:
            avg = sum(column) / len(column)

        for i in missing_rows:
            self.data[i][col_id] = avg

        return self

    def replace_na_with_mean(self, column_name):
        if column_name not in self.column_names:
            raise ValueError(f"Column {column_name} not found!")
        col_idx = self.column_names.index(column_name)

        values = [float(row[col_idx]) for row in self.data if row[col_idx] != "NA"]
        mean_value = sum(values) / len(values) if values else 0

        for row in self.data:
            if row[col_idx] == "NA":
                row[col_idx] = mean_value

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        stats = MyPyTable(['attribute', 'min', 'max', 'mid', 'avg', 'median'])
        for col_name in col_names:
            column = self.get_column(col_name, include_missing_values=False)
            if len(column) == 0:
                continue
            try:

                min_val = min(column)
                max_val = max(column)
                mid_val = (min_val + max_val) / 2
                avg_val = sum(column) / len(column)

                sorted_column = sorted(column)
                n = len(sorted_column)
                if n % 2 == 1:
                    median_val = sorted_column[n // 2]
                else:
                    median_val = (sorted_column[(n // 2) - 1] + sorted_column[n // 2]) / 2

                stats.data.append([col_name, min_val, max_val, mid_val, avg_val, median_val])
            except ValueError as error:
                raise ValueError(f"Error computing summary statistics for column '{col_name}': {error}") from error
        return stats



    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        self_key_indices = []
        other_key_indices = []

        for key in key_column_names:
            self_key_indices.append(self.column_names.index(key))

        for key in key_column_names:
            other_key_indices.append(other_table.column_names.index(key))

        new_column_names = self.column_names[:]
        for i, col in enumerate(other_table.column_names):
            if i not in other_key_indices:
                new_column_names.append(col)

        new_table = MyPyTable(new_column_names)

        for row in self.data:
            self_key_column = []
            for i in self_key_indices:
                self_key_column.append(str(row[i]).strip())
            for other_row in other_table.data:
                other_key_column = []
                for j in other_key_indices:
                    other_key_column.append(str(other_row[j]).strip())

                if self_key_column == other_key_column:
                    new_row = row[:]
                    for j, other_element in enumerate(other_row):
                        if j not in other_key_indices:
                            new_row.append(other_element)
                    new_table.data.append(new_row)

        return new_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """

        def find_key_indices(table, key_column_names):
            """Find indices of key columns for the given table."""
            return [table.column_names.index(key) for key in key_column_names]

        def get_key_column_values(row, key_indices):
            """Extract key column values from the row based on key indices."""
            return [str(row[i]).strip() for i in key_indices]

        def create_new_row(row, other_row, other_key_indices):
            """Create a new row by merging the current row with the other table's row."""
            new_row = row[:]
            for j, other_element in enumerate(other_row):
                if j not in other_key_indices:
                    new_row.append(other_element)
            return new_row

        def add_unmatched_rows(data, used_rows_id, num_non_key_columns):
            """Add rows from the data that are not in used_rows_id."""
            new_rows = []
            for i, row in enumerate(data):
                if i not in used_rows_id:
                    new_row = row[:]
                    new_row.extend(["NA"] * num_non_key_columns)
                    new_rows.append(new_row)
            return new_rows

        def merge_other_table_rows(other_table, other_used_rows_id, key_column_names):
            """Merge rows from the other table that don't match with rows in self."""
            new_rows = []
            for j, other_row in enumerate(other_table.data):
                if j not in other_used_rows_id:
                    new_row = ["NA"] * len(self.column_names)

                    for k in range(len(key_column_names)):
                        self_idx = self_key_indices[k]
                        other_idx = other_key_indices[k]
                        new_row[self_idx] = other_row[other_idx]

                    other_non_key_values = [other_row[i] for i in range(len(other_table.column_names)) if i not in other_key_indices]
                    new_row.extend(other_non_key_values)
                    new_rows.append(new_row)
            return new_rows

        self_key_indices = find_key_indices(self, key_column_names)
        other_key_indices = find_key_indices(other_table, key_column_names)

        new_column_names = self.column_names[:]
        for i, col in enumerate(other_table.column_names):
            if i not in other_key_indices:
                new_column_names.append(col)

        new_table = MyPyTable(new_column_names)

        self_used_rows_id = set()
        other_used_rows_id = set()

        for i, row in enumerate(self.data):
            self_key_column = get_key_column_values(row, self_key_indices)
            for j, other_row in enumerate(other_table.data):
                other_key_column = get_key_column_values(other_row, other_key_indices)
                if self_key_column == other_key_column:
                    self_used_rows_id.add(i)
                    other_used_rows_id.add(j)

                    new_row = create_new_row(row, other_row, other_key_indices)
                    new_table.data.append(new_row)

        num_other_non_key_columns = len(other_table.column_names) - len(other_key_indices)
        new_table.data.extend(add_unmatched_rows(self.data, self_used_rows_id, num_other_non_key_columns))

        new_table.data.extend(merge_other_table_rows(other_table, other_used_rows_id, key_column_names))

        return new_table
