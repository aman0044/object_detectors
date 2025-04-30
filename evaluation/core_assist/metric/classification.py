import json
import math
import os
import random
import traceback
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from core_assist.image_ops.src.image_utils import load_rgb
from core_assist.plot import plot
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score)


class Image:
    def __init__(self, gt_json_path, pred_json_path):
        self.gt_anno = self.load_json(gt_json_path)
        self.pred_anno = self.load_json(pred_json_path)
        self.img_path = self.gt_anno["image_path"]
        self.image_name = self.gt_anno["image_name"]

    def plot(self):
        """
        Displays the image with a title showing ground truth, prediction, and score.

        Raises:
            ValueError: If the image path is not set.
        """
        if self.img_path is None:
            raise ValueError("Image path is not set.")

        img = load_rgb(self.img_path)  # Load the image
        gt_label = self.gt_anno["annotations"].get("label", [None])[0]
        pred_label = self.pred_anno["predictions"].get("label", [None])[0]
        pred_score = self.pred_anno["predictions"].get("score", [None])[0]

        # Create the title using the labels
        label_text = f"GT: {gt_label} | Pred: {pred_label} | Score: {pred_score:.2f}"

        # Pass the image with the label as the title to the `image` function
        plot.image(**{label_text: img})

    def load_json(self, json_path):
        try:
            with open(json_path, "r") as f:
                res = json.load(f)
        except Exception as e:
            raise ValueError(
                f"Error loading JSON file {json_path}: {traceback.format_exc()}"
            )

        return res


class DatasetStats:
    def __init__(
        self,
        gt_json_paths=None,
        pred_json_paths=None,
        classes=None,
        y_true=None,
        y_pred=None,
        pred_score=None,
    ):

        self.all_image_objects, self.dataframe = self.get_image_objs_df(
            gt_json_paths=gt_json_paths,
            pred_json_paths=pred_json_paths,
            y_true=y_true,
            y_pred=y_pred,
            pred_score=pred_score,
        )
        self.y_true = None
        self.y_pred = None
        self.classes = classes
        self.cm = None

    def get_stats(self, averaging_methods=["micro", "macro", "weighted"]):
        """
        Computes precision, recall, F1-score for each class and dataset-wide metrics.

        Args:
            averaging_methods (list): Averaging methods ['micro', 'macro', 'weighted'].

        Updates:
            - self.dataset_result: Pandas DataFrame containing evaluation results.
        """
        self.y_true = self.dataframe["gt_label"].to_list()
        self.y_pred = self.dataframe["pred_label"].to_list()
        # self.classes = np.unique(self.dataframe["pred_label"].to_list())
        self.cm = self.compute_confusion_matrix()

        valid_methods = {"binary", "micro", "macro", "weighted"}
        assert all(
            method in valid_methods for method in averaging_methods
        ), f"Invalid averaging method. Choose from {valid_methods}"

        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None
        )

        per_class_results = pd.DataFrame(
            {
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Support": support.astype(int),  # Ensure support is integer
            },
            index=self.classes,
        )

        print(f"\nPer-Class Metrics:\n{per_class_results.to_markdown()}\n")

        # Compute dataset-wide metrics
        dataset_results = {
            "Precision": [
                precision_score(self.y_true, self.y_pred, average=method)
                for method in averaging_methods
            ],
            "Recall": [
                recall_score(self.y_true, self.y_pred, average=method)
                for method in averaging_methods
            ],
            "F1": [
                f1_score(self.y_true, self.y_pred, average=method)
                for method in averaging_methods
            ],
        }

        self.dataset_result = pd.DataFrame(dataset_results, index=averaging_methods)

        print(f"\nEvaluation Results:\n{self.dataset_result.to_markdown()}\n")

    def compute_confusion_matrix(self):
        """
        Computes the confusion matrix for binary or multiclass classification.

        Returns:
            np.array: Confusion matrix.
        """
        return confusion_matrix(self.y_true, self.y_pred, labels=self.classes)

    def get_image_objs_df(
        self,
        gt_json_paths=None,
        pred_json_paths=None,
        y_true=None,
        y_pred=None,
        pred_score=None,
    ):
        """
        Generates a DataFrame containing image metadata and object annotations in parallel.

        Args:
            gt_json_paths (list, optional): List of paths to ground truth JSON files.
            pred_json_paths (list, optional): List of paths to prediction JSON files.
            y_true (list, optional): List of ground truth labels if JSONs are not provided.
            y_pred (list, optional): List of predicted labels if JSONs are not provided.
            pred_score (list, optional): List of prediction scores if JSONs are not provided.

        Returns:
            tuple: A dictionary mapping image_names to `Image` objects and a pandas DataFrame.
        """
        if gt_json_paths and not isinstance(gt_json_paths, list):
            raise ValueError("gt_json_paths must be a list of file paths or None.")
        if pred_json_paths and not isinstance(pred_json_paths, list):
            raise ValueError("pred_json_paths must be a list of file paths or None.")

        all_image_obj = {}
        results = []

        def process_json(gt_path, pred_path):
            """Reads JSON files and extracts necessary data."""
            if not os.path.exists(gt_path) or not os.path.exists(pred_path):
                return None  # Skip missing files

            with open(gt_path, "r") as f:
                gt_res = json.load(f)
            with open(pred_path, "r") as f:
                pred_res = json.load(f)

            img_obj = Image(gt_path, pred_path)
            all_image_obj.setdefault(img_obj.image_name, []).append(img_obj)

            return {
                "image_path": gt_res.get("image_path"),
                "image_name": gt_res.get("image_name"),
                "gt_label": gt_res.get("annotations", {}).get("label", [None])[0],
                "pred_label": pred_res.get("predictions", {}).get("label", [None])[0],
                "pred_scores": pred_res.get("predictions", {}).get("score", [None])[0],
            }

        # Use ThreadPoolExecutor for parallel file reading
        if gt_json_paths and pred_json_paths:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(
                    executor.map(process_json, gt_json_paths, pred_json_paths)
                )

            # Filter out None values (failed file loads)
            results = [r for r in results if r]

            # Convert to DataFrame efficiently
            df = pd.DataFrame(results)
            return all_image_obj, df

        # Case 2: If no JSONs, use manually provided values
        if y_true is not None and y_pred is not None and pred_score is not None:
            df = pd.DataFrame(
                {"gt_label": y_true, "pred_label": y_pred, "pred_scores": pred_score}
            )
            return {}, df

        # Case 3: If no valid input, return empty results
        return {}, pd.DataFrame()

    def filter_data(self, class_label, metric):
        """
        Filters the dataset based on a specific class and metric.

        Args:
            class_label (str): The class to filter on. Must be in self.classes.
            metric (str): The metric to filter by. Must be one of ['false_positive', 'true_positive', 'false_negative'].

        Returns:
            DatasetStats | pd.DataFrame | None:
                - A new DatasetStats object if image objects exist.
                - A filtered DataFrame if no image objects exist.
                - None if no matching data is found.

        Raises:
            ValueError: If `class_label` is not in `self.classes` or `metric` is invalid.
        """

        # Ensure dataset is initialized
        if self.dataframe is None or self.all_image_objects is None:
            raise ValueError("Dataset is not initialized properly.")

        # Validate class_label
        if class_label not in self.classes:
            raise ValueError(
                f"Invalid class_label: {class_label}. Must be one of {list(self.classes)}"
            )

        # Validate metric
        if metric not in ["false_positive", "true_positive", "false_negative"]:
            raise ValueError(
                "Metric must be one of ['false_positive', 'true_positive', 'false_negative']."
            )

        # Apply filtering based on the metric
        if metric == "false_positive":
            filtered_df = self.dataframe[
                (self.dataframe["pred_label"] == class_label)
                & (self.dataframe["gt_label"] != class_label)
            ]
        elif metric == "true_positive":
            filtered_df = self.dataframe[
                (self.dataframe["pred_label"] == class_label)
                & (self.dataframe["gt_label"] == class_label)
            ]
        elif metric == "false_negative":
            filtered_df = self.dataframe[
                (self.dataframe["pred_label"] != class_label)
                & (self.dataframe["gt_label"] == class_label)
            ]

        # If no matches, return None or an empty DataFrame
        if filtered_df.empty:
            print(
                f"Warning: No data found for class '{class_label}' with metric '{metric}'."
            )
            return None if self.all_image_objects else filtered_df

        # If image objects exist, create a new DatasetStats object
        if self.all_image_objects:
            filtered_img_objs = {
                img_name: self.all_image_objects[img_name]
                for img_name in filtered_df["image_name"].values
                if img_name in self.all_image_objects
            }

            # Create a new DatasetStats object manually
            new_instance = DatasetStats()
            new_instance.all_image_objects = filtered_img_objs
            new_instance.dataframe = filtered_df
            new_instance.y_true = filtered_df["gt_label"].to_list()
            new_instance.y_pred = filtered_df["pred_label"].to_list()

            # Ensure `self.classes` only contains valid labels
            valid_classes = set(new_instance.y_true + new_instance.y_pred)
            new_instance.classes = np.array(
                list(valid_classes)
            )  # Ensure only existing labels are included

            # Compute confusion matrix only if valid classes exist
            if new_instance.classes.size > 0:
                new_instance.cm = new_instance.compute_confusion_matrix()
            else:
                new_instance.cm = None  # No confusion matrix for empty class set

            return new_instance

        # If no image objects, return only the filtered DataFrame
        return filtered_df

    def plot(self, samples=5, rgb_flag=False):
        """
        Plots a given number of sample images in a grid layout using `display_multiple_images()`.

        Args:
            all_image_objects (dict): Dictionary of image objects.
            samples (int): Number of images to plot. Defaults to 5.
            rgb_flag (bool): If True, converts BGR images to RGB before displaying.

        Raises:
            ValueError: If `all_image_objects` is empty.
        """
        if not self.all_image_objects:
            raise ValueError("No image objects available for plotting.")

        # Flatten all image objects into a list
        all_images = [
            img_obj
            for img_list in self.all_image_objects.values()
            for img_obj in img_list
        ]

        # Adjust sample size if needed
        samples = min(samples, len(all_images))

        # Randomly select `samples` number of images
        selected_images = random.sample(all_images, samples)

        # Extract image paths and labels
        image_paths = [img_obj.img_path for img_obj in selected_images]
        labels = [
            f"GT: {img_obj.gt_anno['annotations'].get('label', [None])[0]}\n"
            f"Pred: {img_obj.pred_anno['predictions'].get('label', [None])[0]} "
            f"({img_obj.pred_anno['predictions'].get('score', [None])[0]:.2f})"
            for img_obj in selected_images
        ]

        # Compute optimal grid layout (rows & cols)
        cols = min(5, samples)  # Limit to max 5 columns
        rows = math.ceil(samples / cols)

        # Call display function
        plot.display_multiple_images(
            image_paths, labels=labels, rows=rows, cols=cols, rgb_flag=rgb_flag
        )

    def _plot_confusion_matrix(self, normalize=True, cmap="Blues"):
        """
        Plots a confusion matrix using Seaborn's heatmap.

        Args:
            normalize (bool): If True, plots a percentage-wise confusion matrix; otherwise, plots raw counts.
            cmap (str): Colormap for the heatmap (default is 'Blues').

        Raises:
            ValueError: If confusion matrix is not computed or empty.
        """
        if self.cm is None or self.cm.size == 0:
            raise ValueError("Confusion matrix is empty. Ensure data is available.")

        # Compute percentage-wise confusion matrix if normalize is True
        if normalize:
            cm_display = np.array(
                [row / np.sum(row) if np.sum(row) > 0 else row for row in self.cm]
            )
            fmt = ".2%"  # Format for percentage
            title = "Confusion Matrix (Percentage)"
        else:
            cm_display = self.cm
            fmt = "d"  # Format for integer counts
            title = "Confusion Matrix (Raw Counts)"

        # Convert to DataFrame for plotting
        df_cm = pd.DataFrame(cm_display, index=self.classes, columns=self.classes)

        # Plot the heatmap
        plt.figure(figsize=(12, 7))
        sns.heatmap(df_cm, annot=True, fmt=fmt, cmap=cmap)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
        plt.show()

    def plot_metric(
        self, metric_type="confusion_matrix", cmap="Blues", normalize=False
    ):
        """Plots either the confusion matrix or a violin plot of predicted scores.

            This function provides a way to visualize evaluation metrics. It can display
            either a confusion matrix (showing classification performance) or a violin plot
            of predicted scores (showing the distribution of scores for each class).

        Args:
            metric_type (str): Type of metric to plot. Must be "confusion_matrix" or "score".
            cmap (str): Colormap for the heatmap (default is 'Blues'). Only used if metric_type is "confusion_matrix".
            normalize (bool): If True, normalizes the confusion matrix. Only used if metric_type is "confusion_matrix".

        Raises:
            ValueError: If metric_type is not "confusion_matrix" or "score".
        """

        if metric_type == "confusion_matrix":
            self._plot_confusion_matrix(normalize=normalize, cmap=cmap)
        elif metric_type == "score":
            if not hasattr(self, "dataframe"):  # Check if dataframe exists
                raise ValueError(
                    "Dataframe is required for score plot. Ensure it's available."
                )
            if self.dataframe.empty:
                raise ValueError("Dataframe is empty. Ensure data is available.")

            fig, ax = plt.subplots(figsize=(10, 5))  # Create figure and axes

            sns.violinplot(
                x=self.dataframe["pred_scores"], y=self.dataframe["pred_label"], ax=ax
            )  # Fixed column names
            ax.set_xlabel("Predicted Label", fontsize=14)  # Increased font size
            ax.set_ylabel("Predicted Scores", fontsize=14)  # Increased font size
            ax.set_title(
                "Predicted Scores by Label", fontsize=16
            )  # Increased font size
            plt.xticks(
                rotation=45, ha="right", fontsize=12
            )  # Rotated x-axis labels for readability, increased font size
            plt.yticks(fontsize=12)  # Increased font size
            plt.tight_layout()  # Adjust layout to prevent labels from overlapping
            plt.show()
        else:
            raise ValueError(
                "Invalid metric_type. Must be 'confusion_matrix' or 'score'."
            )

    def compute_confusion_matrix(self):
        """
        Computes the confusion matrix for binary or multiclass classification.

        Returns:
            np.array: Confusion matrix.
        """
        return confusion_matrix(self.y_true, self.y_pred, labels=self.classes)

    def accuracy(self):
        """
        Computes the accuracy score.

        Returns:
            float: Accuracy score.
        """
        return np.mean(np.array(self.y_true) == np.array(self.y_pred))

    def precision(self, average="binary"):
        """
        Computes the precision score.

        Parameters:
            average (str): Averaging method ('binary', 'macro', 'micro', 'weighted').

        Returns:
            float: Precision score.

        Raises:
            ValueError: If invalid average is provided.
        """
        if average not in ["binary", "macro", "micro", "weighted"]:
            raise ValueError(
                "Invalid value for average. Choose from 'binary', 'macro', 'micro', 'weighted'."
            )

        if average == "binary":
            tp = self.cm[1, 1]
            fp = self.cm[0, 1]
            return tp / (tp + fp) if tp + fp > 0 else 0
        elif average in ["macro", "weighted"]:
            precisions = []
            for i in range(len(self.cm)):
                tp = self.cm[i, i]
                fp = np.sum(self.cm[:, i]) - tp
                precisions.append(tp / (tp + fp) if tp + fp > 0 else 0)
            return (
                np.mean(precisions)
                if average == "macro"
                else np.average(precisions, weights=np.sum(self.cm, axis=1))
            )
        elif average == "micro":
            tp = np.trace(self.cm)
            fp = np.sum(self.cm) - tp
            return tp / (tp + fp) if tp + fp > 0 else 0

    def recall(self, average="binary"):
        """
        Computes the recall score.

        Parameters:
            average (str): Averaging method ('binary', 'macro', 'micro', 'weighted').

        Returns:
            float: Recall score.

        Raises:
            ValueError: If invalid average is provided.
        """
        if average not in ["binary", "macro", "micro", "weighted"]:
            raise ValueError(
                "Invalid value for average. Choose from 'binary', 'macro', 'micro', 'weighted'."
            )

        if average == "binary":
            tp = self.cm[1, 1]
            fn = self.cm[1, 0]
            return tp / (tp + fn) if tp + fn > 0 else 0
        elif average in ["macro", "weighted"]:
            recalls = []
            for i in range(len(self.cm)):
                tp = self.cm[i, i]
                fn = np.sum(self.cm[i, :]) - tp
                recalls.append(tp / (tp + fn) if tp + fn > 0 else 0)
            return (
                np.mean(recalls)
                if average == "macro"
                else np.average(recalls, weights=np.sum(self.cm, axis=1))
            )
        elif average == "micro":
            tp = np.trace(self.cm)
            fn = np.sum(self.cm) - tp
            return tp / (tp + fn) if tp + fn > 0 else 0

    def f1_scores(self, average="binary"):
        """
        Computes the F1 score based on the given confusion matrix.

        Parameters:
            conf_matrix (np.array): Confusion matrix (square matrix for multi-class).
            average (str): Averaging method ('binary', 'macro', 'micro', 'weighted').

        Returns:
            float: F1 score.

        Raises:
            ValueError: If invalid average is provided.
        """
        if average not in ["binary", "macro", "micro", "weighted"]:
            raise ValueError(
                "Invalid value for average. Choose from 'binary', 'macro', 'micro', 'weighted'."
            )

        conf_matrix = self.cm

        # True Positives, False Positives, False Negatives
        TP = np.diag(conf_matrix)
        FP = np.sum(conf_matrix, axis=0) - TP
        FN = np.sum(conf_matrix, axis=1) - TP
        support = np.sum(conf_matrix, axis=1)  # Total actual instances per class

        # Precision & Recall (handle division by zero)
        precision = np.where((TP + FP) > 0, TP / (TP + FP), 0)
        recall = np.where((TP + FN) > 0, TP / (TP + FN), 0)

        # F1-score for each class
        f1_per_class = np.where(
            (precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0
        )

        if average == "macro":
            return np.mean(f1_per_class)

        elif average == "micro":
            total_TP = np.sum(TP)
            total_FP = np.sum(FP)
            total_FN = np.sum(FN)
            precision_micro = (
                total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
            )
            recall_micro = (
                total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
            )
            return (
                2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
                if (precision_micro + recall_micro) > 0
                else 0
            )

        elif average == "weighted":
            return np.sum(f1_per_class * support) / np.sum(support)

        elif average == "binary":
            if len(TP) != 2:
                raise ValueError(
                    "Binary F1-score can only be computed for a 2-class confusion matrix."
                )
            return f1_per_class[1]  # Assuming the second class is the positive class

    def false_positives(self):
        """
        Computes the number of false positives for each class.

        Returns:
            np.array: False positives for each class.
        """
        return np.sum(self.cm, axis=0) - np.diag(self.cm)

    def false_negatives(self):
        """
        Computes the number of false negatives for each class.

        Returns:
            np.array: False negatives for each class.
        """
        return np.sum(self.cm, axis=1) - np.diag(self.cm)

    def true_positives(self):
        """
        Computes the number of true positives for each class.

        Returns:
            np.array: True positives for each class.
        """
        return np.diag(self.cm)

    def true_negatives(self):
        """
        Computes the number of true negatives for each class.

        Returns:
            np.array: True negatives for each class.
        """
        total = np.sum(self.cm)
        tp_fp_fn = np.sum(self.cm, axis=1) + np.sum(self.cm, axis=0) - np.diag(self.cm)
        return total - tp_fp_fn
