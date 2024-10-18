"""
main.py

This main file is to run the entire process of generating counterfactuals for a given dataset and class label.
Note: modify the get_dataset_classifier_mappings function to run the code for different datasets and class labels.
      include the following format of the information: (dataset_name, class_label, saved_model_file); it can run only for one dataset and class label at a time.

Last updated: 10/18/2024
"""

import timeit
import numpy as np
import random

from DBSCANClustering import DBSCANClustering
from KMeansClustering import KMeansClustering
from RangeCalculator import RangeCalculator
from Helper import PermittedRangeManager
from CfGenerator import CfGenerator
from DataProcessing import DataProcessing
from ModelTrainer import ModelTrainer

np.random.seed(42)
random.seed(42)


def get_feature_sets():
    feature_set1 = [f"x{i}" for i in range(1, 21)]
    feature_set2 = [f"x{i}" for i in range(21, 35)]
    all_features = feature_set1 + feature_set2
    # return feature sets along with their names
    return [
        (feature_set1, "feature_set1"),
        (feature_set2, "feature_set2"),
        (all_features, "all_features"),
    ]


def get_dataset_classifier_mappings():
    return [
        # use the best performing model for each dataset and each class label
        # contain (dataset_name, class_label, saved_model_file)
        # comment out one of the following each time to run the code
        #("run1_data.csv", "c2", "run1_data_c2_mlp_classifier.pkl"),
        ("run1_data.csv", "c4", "run1_data_c4_mlp_classifier.pkl"),
    ]


def main():
    start = timeit.default_timer()
    feature_sets = get_feature_sets()
    dataset_classifier_mappings = get_dataset_classifier_mappings()

    for dataset, class_label, _ in dataset_classifier_mappings:
        extracted_data_name = dataset.split(".")[0]
        print(f"{extracted_data_name} - {class_label}")

        # data processing
        data_processor = DataProcessing(dataset, class_label)

        # train or load models
        model_trainer = ModelTrainer(
            data_processor.data, class_label, extracted_data_name
        )
        model_trainer.train_and_save_models()

        # initialize DBSCANClustering class
        dbscan_clustering = DBSCANClustering(
            data_processor.data, data_processor.data_wo_label, class_label
        )

        # perform DBSCAN Clustering with eps values from 0.4 to 0.8
        eps = 0.4
        for i in range(4, 9):  # i will be used for the file name
            dbscan_clustering.perform_dbscan(eps, i, class_label, extracted_data_name)
            eps += 0.1

        # initialize KMeansClustering class
        kmeans_clustering = KMeansClustering(
            data_processor.data, data_processor.data_wo_label, class_label
        )

        # perform KMeans Clustering with k values from 3 to 5
        for k in range(3, 6):
            kmeans_clustering.perform_kmeans_clustering(
                k, extracted_data_name, class_label
            )

        # perform KMeans clustering with k=2
        updated_df, results = kmeans_clustering.perform_kmeans_clustering(
            2, extracted_data_name, class_label
        )

        # re-initialize kmeansclustering using the updated df
        kmeans = KMeansClustering(updated_df, data_processor.data_wo_label, class_label)
        # calculate kmeans distances for counterfactual generation
        results = kmeans.calculate_distance(class_label)
        sample_index = results["C1"]["sample_index"]

        # generate distance plots for each 'C1' and 'C2'
        for cluster_label in ["C1", "C2"]:
            ordered_df = results[cluster_label]["ordered_df"]
            plot_name = f"{extracted_data_name}_{class_label}_{cluster_label}_distance_plot_{sample_index}.png"
            kmeans.generate_distance_plot(
                ordered_df, class_label, cluster_label, plot_name
            )

        # set fixed permitted ranges
        fixed_permitted_ranges = PermittedRangeManager.predefined_ranges(dataset)

        # log the volume for the fixed permitted ranges
        range_calculator = RangeCalculator(data_processor.data)
        original_volume_description = (
            f"original volume for {extracted_data_name} - {class_label}"
        )
        range_calculator.calculate_and_log_volume(
            fixed_permitted_ranges, original_volume_description, "volume_log.txt"
        )

        # calculate new permitted ranges within cluster C1
        new_min_max_pairs = range_calculator.find_min_max_for_cluster("C1", class_label)
        new_permitted_ranges = range_calculator.calculate_permitted_ranges(
            new_min_max_pairs
        )

        # log the volume for the new permitted ranges
        new_volume_description = f"new volume for {extracted_data_name} - {class_label}"
        range_calculator.calculate_and_log_volume(
            new_permitted_ranges, new_volume_description, "volume_log.txt"
        )

        # initialize CfGenerator class with fixed permitted ranges
        cf_generator = CfGenerator(
            model_name=f"{extracted_data_name}_{class_label}_mlp_classifier.pkl",
            permitted_ranges=fixed_permitted_ranges,
            data=data_processor.data,
            target_class=class_label,
            sample=data_processor.data.loc[[sample_index]],
            sample_index=sample_index,
        )

        # generate counterfactuals using fixed permitted ranges
        cf_generator.generate_cfs_for_feature_sets(feature_sets, "dice_results")

        # update the permitted ranges using the same CfGnerator instance
        #cf_generator.permitted_ranges = new_permitted_ranges

        # initialize CfGenerator class with updated permitted ranges
        cf_generator = CfGenerator(
            model_name=f"{extracted_data_name}_{class_label}_mlp_classifier.pkl",
            permitted_ranges=new_permitted_ranges,
            data=data_processor.data,
            target_class=class_label,
            sample=data_processor.data.loc[[sample_index]],
            sample_index=sample_index,
        )

        # generate counterfactuals using updated permitted ranges
        cf_generator.generate_cfs_for_feature_sets(feature_sets, "results")

    stop = timeit.default_timer()
    time_seconds = stop - start
    time_minutes = time_seconds / 60
    print(f"Time: {time_minutes} minutes")


if __name__ == "__main__":
    main()
