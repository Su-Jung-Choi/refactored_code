"""
RangeCalculator.py

This module is responsible for calculating the range of values for each feature within a specified cluster.
The specified cluster is the cluster that is correctly classified as good design.
The range of values is found to set new permitted ranges for generating counterfactuals.

Last updated: 10/18/2024
"""

import numpy as np
import random

np.random.seed(42)
random.seed(42)


class RangeCalculator:
    def __init__(self, df):
        self.df = df

    def find_min_max_for_cluster(self, cluster_label, label):
        """calculate the min and max values for each feature within the specified cluster"""
        cluster_data = self.df[self.df["cluster"] == cluster_label]

        # separate the data by labels, 0 (good design) and 1 (bad design)
        label0_data = cluster_data[cluster_data[label] == 0].drop(
            columns=[label, "cluster"]
        )

        # calculate min and max values for label 0
        label0_min_values = label0_data.min().tolist()  # change to list
        label0_max_values = label0_data.max().tolist()

        # pair the min and max values for each feature column
        label0_min_max_pairs = [
            [min_val, max_val]
            for min_val, max_val in zip(label0_min_values, label0_max_values)
        ]
        return label0_min_max_pairs

    def calculate_permitted_ranges(self, min_max_pairs):
        permitted_ranges = {
            f"x{i+1}": min_max_pairs[i] for i in range(len(min_max_pairs))
        }
        return permitted_ranges

    def calculate_volume(self, min_max_pairs):
        """
        calculate the volume of the search space based on min and max values.
        The volume is calculated as the product of the difference between the upper bound and lower bound for each feature.
        """
        volume = 1
        for pair in min_max_pairs:
            volume *= pair[1] - pair[0]
        return volume

    def log_volume(self, volume, description, file_path):
        with open(file_path, "a") as file:

            file.write(f"{description}: {volume}\n")
        print(f"{description}: {volume}")

    def calculate_and_log_volume(self, permitted_ranges, description, file_path):
        min_max_pairs = [[bounds[0], bounds[1]] for bounds in permitted_ranges.values()]
        volume = self.calculate_volume(min_max_pairs)
        self.log_volume(volume, description, file_path)
        return volume
