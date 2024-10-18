"""
DataProcessing.py

This module is responsible for reading the data from the file and selecting the target class.

Last updated: 10/18/2024
"""

import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)


class DataProcessing:
    def __init__(self, file_name, target_class):
        self.file_name = file_name
        self.target_class = target_class
        self.data, self.data_wo_label = self._read_data()

    def _read_data(self):
        # Read the first row of the file to know the number of columns
        first_row = pd.read_csv(self.file_name, nrows=1, index_col=0)
        num_columns = len(first_row.columns)

        # Read the data into pandas df, set appropriate column names, and drop the first index column
        data = pd.read_csv(
            self.file_name,
            names=[f"c{i+1}" if i < 4 else f"x{i-3}" for i in range(num_columns)],
            index_col=0,
        )

        # For classification labels 'c2', 'c3', 'c4', replace all values greater than 0 with 1
        for label in ["c2", "c3", "c4"]:
            data[label] = data[label].apply(lambda x: 1 if x > 0 else 0)

        data_wo_label = data.drop(columns=["c1", "c2", "c3", "c4"])

        return self._select_target_class(data), data_wo_label

    def _select_target_class(self, data):
        if self.target_class not in data.columns:
            raise ValueError(f"Target class {self.target_class} not found in the data.")

        return data.drop(
            columns=[
                col
                for col in data.columns
                if col.startswith("c") and col != self.target_class
            ]
        )

