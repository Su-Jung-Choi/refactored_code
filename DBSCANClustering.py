"""
DBSCANClustering.py

This module is responsible for performing DBSCAN clustering on the data and visualizing the results.

Last updated: 10/18/2024
"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(42)
random.seed(42)


class DBSCANClustering:
    def __init__(self, df_with_label, df_wo_label, col_name):
        self.df_with_label = df_with_label
        self.df_wo_label = df_wo_label
        self.col_name = col_name

    def compute_dbscan(self, eps=0.5, min_samples=5):
        # standardize the features
        X = StandardScaler().fit_transform(self.df_wo_label)

        # reduce dimensionality using PCA
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

        # apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X)

        # extract the target label column
        label = self.df_with_label[self.col_name]

        # add the cluster labels to the dataframe
        self.df_with_label["dbscan"] = dbscan_labels

        # map cluster labels
        unique_clusters = sorted(set(dbscan_labels))
        cluster_mapping = {
            cluster: f"C{i}"
            for i, cluster in enumerate(unique_clusters)
            if cluster != -1
        }
        cluster_mapping[-1] = " noise"
        self.df_with_label["dbscan"] = self.df_with_label["dbscan"].map(cluster_mapping)

        # group by clusters and count good and bad in each cluster
        cluster_label_counts = (
            self.df_with_label.groupby(["dbscan", label]).size().unstack(fill_value=0)
        )

        # rename the 0 (good design) and 1 (bad design) labels into 'good' and 'bad', respectively
        cluster_label_counts = cluster_label_counts.rename(
            columns={0: "Good", 1: "Bad"}
        )

        return X, cluster_label_counts, dbscan_labels

    def perform_dbscan(self, eps, eps_num, class_label, dataset_name):
        # perform DBSCAN and return results
        X, cluster_label_counts, dbscan_labels = self.compute_dbscan(eps)

        # visualize the clusters
        self.plot_dbscan_scatter(X, dbscan_labels, dataset_name, class_label, eps_num)
        self.plot_dbscan_results(
            cluster_label_counts, dataset_name, class_label, eps_num
        )
        return X, dbscan_labels

    @staticmethod
    def plot_dbscan_scatter(X, dbscan_labels, dataset_name, class_label, eps_num):
        plt.figure(figsize=(6, 4))
        unique_labels = np.unique(dbscan_labels)

        # plot each cluster with a different color
        for label in unique_labels:
            if label == -1:
                color = "k"  # black color for noise
                marker = "x"
                label_text = "Noise"
            else:
                color = plt.cm.Spectral(float(label) / len(unique_labels))
                marker = "o"
                label_text = f"C{label + 1}"
            plt.scatter(
                X[dbscan_labels == label][:, 0],
                X[dbscan_labels == label][:, 1],
                c=[color],
                label=label_text,
                marker=marker,
                s=50,
                edgecolor="k",
            )

        plt.title("DBSCAN Clustering", fontsize=16, fontweight="bold")
        plt.legend(loc="lower left", fontsize=12)

        # save the figure to the 'cluster' folder
        if not os.path.exists("cluster"):
            os.makedirs("cluster")

        plot_filename = os.path.join(
            "cluster", f"{dataset_name}_{class_label}_DBSCAN_eps{eps_num}_scatter.png"
        )
        plt.savefig(plot_filename)
        plt.close()

    @staticmethod
    # plot the DBSCAN results and save each plot in the 'cluster' folder
    def plot_dbscan_results(cluster_label_counts, dataset_name, label_name, eps_num):
        # create a new figure
        plt.figure(figsize=(6, 4))
        colors = ["blue", "red"]

        # visualize the bar chart
        ax = cluster_label_counts.plot(kind="bar", color=colors)
        plt.xlabel("DBSCAN Clustering", fontsize=14, fontweight="bold")
        plt.ylabel("Count", fontsize=14, fontweight="bold")
        plt.xticks(
            range(len(cluster_label_counts)), cluster_label_counts.index, rotation=0
        )
        # Customize the legend with labels and color names
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["Good: Blue", "Bad: Red"], title="", fontsize=12)

        # save the figure to the 'cluster' folder
        if not os.path.exists("dbscan"):
            os.makedirs("dbscan")

        # change the k name according to the number of clusters
        plot_filename = os.path.join(
            "dbscan", f"{dataset_name}_{label_name}_DBSCAN_eps{eps_num}_plot.png"
        )
        plt.savefig(plot_filename)
        plt.close()
