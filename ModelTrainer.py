"""
ModelTrainer.py

This module is responsible for training and saving ML models for a given dataset and class label.

Last updated: 10/18/2024
"""

import os
import numpy as np
import random
import pickle
import csv

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


np.random.seed(42)
random.seed(42)


# Utility function to save results to CSV
def save_to_csv(file_name, results):
    # check if the file already exists
    file_exists = os.path.isfile(file_name)

    with open(file_name, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # write the header only if the file doesn't exist
        if not file_exists:
            writer.writerow(["file name", "label", "model", "accuracy"])

        # write the results
        print("Results:", results)
        for result in results:
            writer.writerow(result)


# ModelTrainer class
class ModelTrainer:
    def __init__(self, dataset, class_label, dataset_name):
        self.dataset = dataset
        self.class_label = class_label
        self.dataset_name = dataset_name
        self.results = []

    def train_models(self):
        print(f"Training models for {self.dataset_name} - {self.class_label}")
        X = self.dataset.drop(columns=[self.class_label])
        y = self.dataset[self.class_label]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define the models
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "DTC": DecisionTreeClassifier(max_depth=25, random_state=42),
            "SVC": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
            "MLP": make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(100, 100, 100), random_state=42),
            ),
            "NBC": GaussianNB(),
        }

        trained_models = {}
        for model_name, model in models.items():
            # Train the model and save accuracy
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            full_model_name = f"{self.dataset_name}_{self.class_label}_{model_name.lower()}_classifier.pkl"
            print(f"{full_model_name} Test Set Accuracy: {accuracy}")
            self.results.append(
                [self.dataset_name, self.class_label, model_name, accuracy]
            )
            trained_models[model_name] = model

        return trained_models

    def save_models(self, models, dataset_name):
        # Save each trained model to a .pkl file
        for model_name, model in models.items():
            model_filename = (
                f"{dataset_name}_{self.class_label}_{model_name.lower()}_classifier.pkl"
            )
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

    def load_models(self, dataset_name):
        # Load the models if they exist
        loaded_models = {}
        for model_name in ["KNN", "DTC", "SVC", "MLP", "NBC"]:
            model_filename = (
                f"{dataset_name}_{self.class_label}_{model_name.lower()}_classifier.pkl"
            )
            if os.path.exists(model_filename):
                with open(model_filename, "rb") as f:
                    loaded_models[model_name] = pickle.load(f)
            else:  # If any model is missing, return None
                print(f"Model file {model_filename} is missing.")
                return None

        return loaded_models

    def get_results(self):
        return self.results

    def train_and_save_models(self):
        models = self.load_models(self.dataset_name)

        if models is None or not self.get_results():
            print(f"Training models for {self.dataset_name} - {self.class_label}...")
            models = self.train_models()
            self.save_models(models, self.dataset_name)
            print("Training is completed.")
        else:
            print(
                f"Loading pre-trained models for {self.dataset_name} - {self.class_label}..."
            )

        save_to_csv("model_performance.csv", self.get_results())
