"""
Helper.py

This helper file contains the predefined ranges for the features in the dataset.

Last updated: 10/18/2024
"""

import numpy as np
import random

np.random.seed(42)
random.seed(42)


class PermittedRangeManager:
    @staticmethod
    def predefined_ranges(file_name):
        permitted_ranges = {}
        # Fixed ranges for x1 to x20
        for i in range(1, 21):
            permitted_ranges[f"x{i}"] = [0, 100] if i % 2 != 0 else [0, 43]
        # Dataset-specific ranges for x21 to x34
        for i in range(21, 35):
            if file_name == "run1_data.csv":
                permitted_ranges[f"x{i}"] = [0.5, 1.5]
            elif file_name == "run2_data.csv":
                permitted_ranges[f"x{i}"] = [0.5, 1.2]
            elif file_name == "run3_data.csv":
                permitted_ranges[f"x{i}"] = [0.5, 1.0]
            elif file_name == "run4_data.csv":
                permitted_ranges[f"x{i}"] = [0.3, 0.8]
            else:
                raise ValueError("Invalid file name for fixed permitted ranges.")
        return permitted_ranges
