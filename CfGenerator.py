"""
CfGenerator.py

This module is responsible for generating counterfactuals for a given sample using DiCE library.

Last updated: 10/18/2024
"""

import os
import numpy as np
import random
import pickle

# Sklearn imports
from sklearn.model_selection import train_test_split

# DiCE imports
import dice_ml

np.random.seed(42)
random.seed(42)


class CfGenerator:
    def __init__(
        self,
        model_name,
        permitted_ranges,
        data,
        target_class,
        sample,
        sample_index,
    ):
        self.model_name = model_name
        self.permitted_ranges = permitted_ranges
        self.data = data
        self.target_class = target_class
        self.sample = sample
        self.sample_index = sample_index
        self.data_object = self.create_data_object()

    def create_data_object(self):
        """create a DiCE data object once during data processing."""
        data_wo_sample = self.data.drop(self.sample_index)
        target = data_wo_sample[self.target_class]
        cleaned_data = data_wo_sample.drop(columns=["cluster"])

        # split the data into training and testing sets
        train_data, _, _, _ = train_test_split(
            cleaned_data, target, test_size=0.2, random_state=0, stratify=target
        )
        # specify all continuous features
        continuous_features = [f"x{i}" for i in range(1, 35)]

        # create the data object
        data_object = dice_ml.Data(
            dataframe=train_data,
            continuous_features=continuous_features,
            outcome_name=self.target_class,
        )
        return data_object

    def generate_cf(self, file):
        """generate 10 cfs for each feature set with given sample and log the results to a file"""
        sample = self.sample.drop(columns=[self.target_class, "cluster"])

        try:
            with open(self.model_name, "rb") as f:
                model = pickle.load(f)

            dice_model = dice_ml.Model(model=model, backend="sklearn")

            # create an instance of DiCE class, which is used to generate counterfacutal explanations
            explanation_instance = dice_ml.Dice(
                self.data_object, dice_model, method="genetic"
            )

            # generate 10 cfs that can change the original outcome (1) to desired outcome (0)
            counterfactuals = explanation_instance.generate_counterfactuals(
                sample,
                total_CFs=10,
                desired_class="opposite",
                features_to_vary=self.feature_set,
                permitted_range=self.permitted_ranges,
            )

            # get the counterfactuals as a dataframe
            counterfactuals_df = counterfactuals.cf_examples_list[0].final_cfs_df
            # log the results
            self.log_counterfactuals(counterfactuals_df, file)

        except Exception as e:
            file.write(
                f"Error in generating cfs for {self.feature_set_name}: {str(e)}\n"
            )

    def log_counterfactuals(self, counterfactuals, file):
        """log the cf results to the file"""

        if not counterfactuals.empty:
            # drop the label column and convert the cfs to a list of lists
            cf_wo_label = counterfactuals.drop(columns=[self.target_class])
            cf_list = cf_wo_label.values.tolist()
            extracted_model_name = self.model_name.split(".")[0]
            # write header to file
            file.write(
                f"# {extracted_model_name} - {self.feature_set_name} 10 counterfactuals: \n"
            )

            # log the counterfactuals
            for i, row in enumerate(cf_list):
                comma_separated_row = ",".join(map(str, row))
                file.write(f"[{comma_separated_row}],\n")
            if len(cf_list) < 10:
                file.write(f"# Only {len(cf_list)} counterfactuals were generated.\n")

        else:
            # log when no cfs are found
            file.write(
                f"# No counterfactuals found for {extracted_model_name} - {self.feature_set_name}.\n"
            )

    def generate_cfs_for_feature_sets(
        self,
        feature_sets,
        folder_name,
    ):
        
        extracted_model_name = self.model_name.split(".")[0]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        txt_file_path = os.path.join(
            folder_name,
            f"{extracted_model_name}_cfs_{self.sample_index}.txt",
        )

        # check if the txt file already exists
        if os.path.exists(txt_file_path):
            print(f"File {txt_file_path} already exists, skipping generation.")
            return  # skip the cf generation and exit the function

        # if file doesn't exist, generate cfs for each feature set
        with open(txt_file_path, "w") as file:
            for feature_set, feature_set_name in feature_sets:
                # update the instance attributes to reflect current feature set
                self.feature_set = feature_set
                self.feature_set_name = feature_set_name

                # generate cfs and save them to a file
                self.generate_cf(file)
