"""
KMeansClustering.py

This module is responsible for performing KMeans clustering on the dataset and visualizing the results.
Note: change the sample_index value in the calculate_distance function to generate cfs for different samples.

Last updated: 10/18/2024
"""

import os
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # to create a custom line in a plot

np.random.seed(42)
random.seed(42)


class KMeansClustering:
    def __init__(self, df_with_label, df_wo_label, col_name):
        self.df_with_label = df_with_label
        self.df_wo_label = df_wo_label
        self.col_name = col_name

    def compute_kmeans(self, k):
        self.df_with_label.drop(columns=["dbscan"], errors="ignore", inplace=True)
        X = self.df_wo_label
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)

        # add a new column 'cluster' to the dataframe
        self.df_with_label["cluster"] = kmeans.labels_
        # rename the cluster labels
        self.df_with_label["cluster"] = self.df_with_label["cluster"].apply(
            lambda x: f"C{x + 1}"
        )

        # group by clusters and count 0 and 1 in each cluster
        cluster_label_counts = (
            self.df_with_label.groupby(["cluster", self.col_name])
            .size()
            .unstack(fill_value=0)
        )

        # rename the 0 (good design) and 1 (bad design) labels into 'Good' and 'Bad', respectively
        cluster_label_counts = cluster_label_counts.rename(
            columns={0: "Good", 1: "Bad"}
        )

        return self.df_with_label, cluster_label_counts

    def perform_kmeans_clustering(self, k, dataset_name, class_label):
        updated_df, cluster_label_counts = self.compute_kmeans(k)

        # Visualize clusters
        self.plot_kmeans_results(cluster_label_counts, dataset_name, class_label, k)

        return updated_df, cluster_label_counts

    @staticmethod
    def plot_kmeans_results(cluster_label_counts, dataset_name, label_name, k):
        title_name = dataset_name.split("_")[0] + "_" + label_name
        plt.figure(figsize=(6, 4))
        colors = ["blue", "red"]

        # visualize the bar chart
        ax = cluster_label_counts.plot(kind="bar", color=colors)
        plt.title(f"K = {k}", fontsize=16, fontweight="bold")
        plt.xlabel("KMeans Clustering", fontsize=14, fontweight="bold")
        plt.ylabel("Count", fontsize=14, fontweight="bold")
        plt.xticks(
            range(len(cluster_label_counts)), cluster_label_counts.index, rotation=0
        )

        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["Good: Blue", "Bad: Red"], title="", fontsize=12)

        # save the figure to the 'cluster' folder
        if not os.path.exists("kmeans"):
            os.makedirs("kmeans")

        plot_filename = os.path.join("kmeans", f"{title_name}_k{k}_plot.png")
        plt.savefig(plot_filename)
        plt.close()

    @staticmethod
    def generate_distance_plot(ordered_data, label, cluster_label, plot_name):
        plt.figure(figsize=(8, 6))

        # extract the distance column from the ordered data and convert the df to np array
        ordered_distance = ordered_data["distance"].values

        # extract the actual label column
        ordered_label = ordered_data[[label]]

        # define colors for different labels
        colors = ["blue" if val == 0 else "red" for val in ordered_label[label]]

        # create a bar plot for the distances
        plt.bar(range(len(ordered_distance)), ordered_distance, color=colors)
        plt.title(f"{cluster_label} Distances", fontsize=14, fontweight="bold")

        # create a custom legend
        handles = [
            Line2D(
                [0], [0], color="blue", linestyle="-", linewidth=10, label="Good: Blue"
            ),
            Line2D(
                [0], [0], color="red", linestyle="-", linewidth=10, label="Bad: Red"
            ),
        ]
        plt.legend(handles=handles, fontsize=12)

        # add x-axis and y-axis labels
        plt.xlabel("Number of Samples", fontsize=14, fontweight="bold")
        plt.ylabel("Distance from the Instance", fontsize=14, fontweight="bold")

        # adjust the layout
        plt.tight_layout(pad=2.0)

        # create a directory to save the plot if it doesn't exist
        if not os.path.exists("plots"):
            os.makedirs("plots")

        # save the plot
        output_path = os.path.join("plots", plot_name)
        print(f"Saving plot to {output_path}")
        plt.savefig(output_path)
        plt.close()

    # this function was used initially to get the sample index, but we got specific sample index to generate cfs for our experiments,
    # so for reproducibility, we are using the same sample index to generate cfs
    # def get_sample_index(self, label):
    # filter only cluster1
    # cluster1 = self.df_with_label[self.df_with_label["cluster"] == "C1"]
    # select a random bad design sample (denoted by 1 in the label) within Cluster 1
    # sample_list = cluster1[cluster1[label] == 1].sample(n=20, random_state=42)
    # print("sample list: ", sample_list)
    # ---------------------------------------------------------------------------------
    # change the value of 0 to any number between 0 and 19 to get the index of the sample that can generate the cfs
    # we needed only 10 samples, but we are taking 20 samples in case some samples are not suitable for generating cfs
    # sample_index = sample_list.index[0]
    # print("sample index: ", sample_index)
    # sample = self.df_with_label.loc[[sample_index]]
    # print("sample: ", sample)
    # return sample_index

    def calculate_distance_per_cluster(self, cluster_label, sample, label):
        # Filter the data for the given cluster
        cluster_data = self.df_with_label[
            self.df_with_label["cluster"] == cluster_label
        ]

        # Remove the label and 'cluster' columns before calculating the euclidean distance
        rest_data_wo_label = cluster_data.drop(columns=[label, "cluster"]).values

        sample_wo_label = sample.drop(columns=[label, "cluster"]).values
        print(f"{cluster_label} data without labels: ", rest_data_wo_label)

        # Compute the euclidean distances between the sample and the rest of the values
        distances = np.linalg.norm(rest_data_wo_label - sample_wo_label, axis=1)

        # Add a new column 'distance' into the cluster_data df
        cluster_data["distance"] = distances

        # Separate the data into two groups based on the label and sort them by the distance
        group_blue = cluster_data[cluster_data[label] == 0].sort_values(by="distance")
        group_red = cluster_data[cluster_data[label] == 1].sort_values(by="distance")

        # Concatenate the sorted groups back together
        ordered_df = pd.concat([group_blue, group_red])

        return ordered_df

    def calculate_distance(self, label):
        # filter only cluster1
        cluster1 = self.df_with_label[self.df_with_label["cluster"] == "C1"]

        # get a random sample index from cluster1
        # sample_index = self.get_sample_index(label)

        # we randomly selected 10 sample indices from the cluster1 and generate cfs for each of them
        # below are the specific sample indices we used for the experiments
        # you can change the sample_index value to any of the following indices to generate cfs for that sample
        # when you change the sample_index value, you need to also change the dataset in the main function
        # for c2: 1544, 1349, 1355, 713, 405, 2062, 2325, 724, 1611, 1714
        # for c4: 1244, 541, 525, 585, 1576, 943, 932, 2455, 572, 1170
        sample_index = 525

        # extract the sample
        sample = cluster1.loc[[sample_index]]

        # remove the sample from the cluster1
        cluster1 = cluster1.drop(index=sample_index)

        # get unique cluster labels
        unique_clusters = self.df_with_label["cluster"].unique()

        # dictionary to store results for each cluster
        results = {}

        for cluster_label in unique_clusters:
            ordered_df = self.calculate_distance_per_cluster(
                cluster_label, sample, label
            )
            results[cluster_label] = {
                "ordered_df": ordered_df,
                "sample_index": sample_index,
            }

        return results
