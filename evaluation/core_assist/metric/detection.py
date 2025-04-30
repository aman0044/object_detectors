import contextlib
import io
import json
import random
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core_assist.image_ops.src.image_utils import load_rgb, resize
from core_assist.metric.utils import (calculate_overall_metrics,
                                      class_metric_to_df,
                                      generate_classwise_tabulated_stats,
                                      generate_overall_dataset_tabulated_stats,
                                      overall_metric_to_df, tabulate_map_res,
                                      transform_to_coco)
from core_assist.plot import plot
from joblib import Parallel, delayed
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


class Image:
    def __init__(self, gt_json_path, pred_json_path, plot_format="xyxy"):
        self.f1_score = {}
        self.precision = {}
        self.recall = {}
        self.miou = {}
        self.gt_anno_path = gt_json_path
        self.img_path = None
        self.pred_anno_path = pred_json_path
        self.results = {}
        self.filename = None
        self.plot_format = plot_format

    def plot(self):
        gt, pred = self.load_json()

        self.img_path = gt["image_path"]
        height, width = gt["height"], gt["width"]

        img = load_rgb(self.img_path)
        img = resize(image=img, size=(width, height))

        gt_bboxes = [item["bbox"] for item in gt["annotations"]]
        gt_labels = [item["label"] for item in gt["annotations"]]

        pred_bboxes = [item["bbox"] for item in pred["predictions"]]
        pred_labels = [item["label"] for item in pred["predictions"]]
        pred_confs = [item["score"] for item in pred["predictions"]]

        gt_img = plot.bbox(img, gt_bboxes, gt_labels, ret=True)
        pred_img = plot.bbox(img, pred_bboxes, pred_labels, pred_confs, ret=True)

        plot.image(
            ground_truth=gt_img,
            predictions=pred_img,
        )

    def get_stats(self, iou_thresh=0.5, conf_thresh=[0.5], tabulate=False):
        """
        Collects object detection statistics for a single file.

        Args:
            gt_json_path (str): path to ground truth json path
            pred_json_path (str): path to prediction json path
            iou_thresh (float): IoU threshold for considering a prediction as a True Positive.
            conf_thresh (list): List of confidence score thresholds for filtering predictions.
            tabulate (bool, optional): If True, return the stats in a tabulated form.
        Returns:
            dict: A dictionary containing evaluation metrics for each confidence threshold.
        """

        results = {}
        gt_labels, predictions = self.load_json()

        self.filename = Path(gt_labels["image_path"]).name
        self.img_path = gt_labels["image_path"]

        for conf in conf_thresh:
            class_wise_stats = {}
            filtered_pred = [
                item for item in predictions["predictions"] if item["score"] >= conf
            ]

            gt_classes = set(item["label"] for item in gt_labels["annotations"])
            pred_classes = set(item["label"] for item in filtered_pred)
            unique_classes = gt_classes.union(pred_classes)

            for class_name in unique_classes:
                filtered_gt = [
                    item
                    for item in gt_labels["annotations"]
                    if item["label"] == class_name
                ]
                filtered_class_pred = [
                    item for item in filtered_pred if item["label"] == class_name
                ]

                tp, fp, fn = 0, 0, 0
                iou_list = []
                matched_gt = set()

                for pred_item in filtered_class_pred:
                    pred_bbox = pred_item["bbox"]
                    pred_bbox = [
                        pred_bbox[0],
                        pred_bbox[1],
                        pred_bbox[0] + pred_bbox[2],
                        pred_bbox[1] + pred_bbox[3],
                    ]

                    best_iou = 0
                    best_gt_idx = None

                    for gt_idx, gt_item in enumerate(filtered_gt):
                        if gt_idx in matched_gt:
                            continue

                        gt_bbox = gt_item["bbox"]
                        gt_bbox = [
                            gt_bbox[0],
                            gt_bbox[1],
                            gt_bbox[0] + gt_bbox[2],
                            gt_bbox[1] + gt_bbox[3],
                        ]

                        iou = self.calculate_iou(gt_bbox, pred_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= iou_thresh:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                        iou_list.append(best_iou)
                    else:
                        fp += 1

                fn = len(filtered_gt) - len(matched_gt)
                class_wise_stats[class_name] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "iou": np.mean(iou_list) if iou_list else 0,
                    "gt": len(filtered_gt),
                }

            f1, precision, recall, miou = self.calculate_object_detection_metrics(
                class_wise_stats
            )
            results[conf] = {
                "filename": self.filename,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "miou": miou,
                "class_stats": class_wise_stats,
            }
            self.precision[conf] = precision
            self.recall[conf] = recall
            self.f1_score[conf] = f1
            self.miou[conf] = miou

        self.results = results

        if tabulate == True:
            print(generate_classwise_tabulated_stats(results))

        return self

    def load_json(self):
        if self.gt_anno_path and self.pred_anno_path is None:
            raise ValueError("jsons paths are not set.")
        with open(self.gt_anno_path, "r") as raw_gt:
            gt = json.load(raw_gt)
        with open(self.pred_anno_path, "r") as raw_pred:
            pred = json.load(raw_pred)

        return gt, pred

    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (list): Coordinates of the first box [x1, y1, x2, y2].
            box2 (list): Coordinates of the second box [x1, y1, x2, y2].

        Returns:
            float: IoU value.
        """
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

    def calculate_object_detection_metrics(self, class_wise_stats):
        """
        Calculate precision, recall, F1-score, and mean IoU from class-wise statistics.

        Args:
            class_wise_stats (dict): Dictionary containing class-wise statistics.

        Returns:
            tuple: A tuple containing F1-score, precision, recall, and mean IoU.
        """
        tp = sum([stats["tp"] for stats in class_wise_stats.values()])
        fp = sum([stats["fp"] for stats in class_wise_stats.values()])
        fn = sum([stats["fn"] for stats in class_wise_stats.values()])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        miou = (
            np.mean([stats["iou"] for stats in class_wise_stats.values()])
            if class_wise_stats
            else 0
        )

        return f1, precision, recall, miou


class DatasetStats:
    def __init__(self, gt_file_paths, pred_file_paths):
        self.gt_file_paths = gt_file_paths
        self.pred_file_paths = pred_file_paths
        self.all_image_stats = {}
        self.class_vise_res_df = None
        self.overall_res_df = None
        self.mAP_df = None
        self.conf_thresh = None

    def get_stats(self, conf_thresh=[0.5], iou_thresh=0.3):
        """
        Compute overall precision, recall, F1-score across multiple files using parallel processing with joblib.

        Args:
            conf_thresh (list): List of confidence thresholds to evaluate.
            iou_thresh (float): IoU threshold for considering a detection as a True Positive.

        Returns:
            DatasetStats object with computed results.
        """
        self.conf_thresh = conf_thresh
        if len(self.gt_file_paths) != len(self.pred_file_paths):
            raise ValueError(
                "Mismatch in the number of ground truth files and prediction files."
            )

        all_class_stats = []
        lock = threading.Lock()
        start_time = time.time()

        def process_file(gt_path, pred_path):
            """
            Process a single file pair and compute statistics.
            """
            img_obj = Image(gt_path, pred_path).get_stats(
                conf_thresh=conf_thresh, iou_thresh=iou_thresh
            )

            with lock:
                filename = img_obj.filename
                if filename in self.all_image_stats:
                    self.all_image_stats[filename].append(img_obj)
                else:
                    self.all_image_stats[filename] = [img_obj]
                all_class_stats.append(img_obj.results)

        # Use joblib for parallel processing

        with tqdm(total=len(self.gt_file_paths), desc="Processing files") as pbar:
            Parallel(n_jobs=-1, backend="threading")(
                delayed(lambda x: (process_file(*x), pbar.update(1)))(
                    (gt_path, pred_path)
                )
                for gt_path, pred_path in zip(self.gt_file_paths, self.pred_file_paths)
            )
        # Convert dataset to COCO format for mAP calculation
        coco_gtr, label_mapping = transform_to_coco(
            self.gt_file_paths, task="detection", is_gt=True
        )
        coco_pred = transform_to_coco(
            self.pred_file_paths,
            task="detection",
            is_gt=False,
            label_mapping=label_mapping,
        )

        # Initialize COCO object and load predictions

        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO()
            coco_gt.dataset = coco_gtr
            coco_gt.createIndex()

            coco_preds = coco_gt.loadRes(coco_pred)

            # Evaluate predictions using COCO metrics
            coco_eval = COCOeval(coco_gt, coco_preds, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        # Compute overall dataset statistics
        tabulated_mAP, mAP_df = tabulate_map_res(coco_eval=coco_eval)
        class_wise_stats, dataset_stats = calculate_overall_metrics(all_class_stats)
        print(generate_overall_dataset_tabulated_stats(class_wise_stats, dataset_stats))
        print(tabulated_mAP)

        # Create DataFrame for mAP results
        elapsed_time = time.time() - start_time
        print(f"Total time taken: {elapsed_time:.2f} seconds")

        # Convert class-level and overall metrics into DataFrames
        self.class_vise_res_df = class_metric_to_df(class_wise_stats)
        self.overall_res_df = overall_metric_to_df(dataset_stats)
        self.mAP_df = mAP_df

        return self

    def filter(self, by="f1_score", thresh=0.5, eq="<", conf_thresh=None):
        """
        Filter the dataset based on a specific metric and threshold.

        Args:
            by (str): The metric to filter by (e.g., "filename, f1_score", "precision", "recall", "miou").
            thresh (float): The threshold value for the metric.
            eq (str): The comparison operator ("<", "<=", ">", ">=", "==").
            conf_thresh (float): The confidence threshold to filter.

        Returns:
            DatasetStats: A new DatasetStats object containing only the filtered images.
        """
        if conf_thresh is None:
            if len(self.conf_thresh) == 1:
                conf_thresh = self.conf_thresh[0]
            else:
                raise ValueError(f"Provide conf_thresh among {self.conf_thresh}")

        if conf_thresh not in self.conf_thresh:
            raise ValueError(f"Provide conf_thresh among {self.conf_thresh}")

        if not self.all_image_stats:
            return self

        new_image_stats = {}

        for filename, img_list in self.all_image_stats.items():
            filtered_list = []
            for img_obj in img_list:
                metric_value = getattr(img_obj, by, None)
                if metric_value is None:
                    raise ValueError(
                        f"Invalid metric '{by}'. Available metrics are 'f1_score', 'precision', 'recall', 'miou'."
                    )

                metric_value = metric_value[conf_thresh]
                if (
                    (eq == "<" and metric_value < thresh)
                    or (eq == "<=" and metric_value <= thresh)
                    or (eq == ">" and metric_value > thresh)
                    or (eq == ">=" and metric_value >= thresh)
                    or (eq == "==" and metric_value == thresh)
                ):
                    filtered_list.append(img_obj)

            if filtered_list:
                new_image_stats[filename] = filtered_list

        # Create a new DatasetStats object with the filtered images
        filtered_dataset = DatasetStats([], [])
        filtered_dataset.all_image_stats = new_image_stats
        filtered_dataset.class_vise_res_df = self.class_vise_res_df
        filtered_dataset.overall_res_df = self.overall_res_df
        filtered_dataset.mAP_df = self.mAP_df

        return filtered_dataset

    def plot(self, samples=3, show_stats=True):
        """
        Visualize ground truth and predicted bounding boxes for a subset of images.

        Args:
            samples (int): Number of images to plot. Defaults to 3.
        """
        if not self.all_image_stats:
            print("No image statistics available to plot.")
            return

        # Flatten the dictionary values (lists) into a single list
        all_images = [
            img_obj
            for img_list in self.all_image_stats.values()
            for img_obj in img_list
        ]

        # Select random images if samples < total available images
        if samples < len(all_images):
            selected_images = random.sample(all_images, samples)
        else:
            selected_images = all_images  # Plot all images if samples exceed available

        # Plot the selected images
        for i, img_obj in enumerate(selected_images, start=1):
            print(f"Plotting image {i}: {img_obj.img_path}")
            if show_stats:
                print(generate_classwise_tabulated_stats(img_obj.results))
            img_obj.plot()

    def __len__(self):
        return len(self.all_image_stats)

    def plot_metric(self, metric_type="all"):
        """
        Plots the specified evaluation metric(s) against the confidence threshold.

        Args:
            metric_type (str, optional): The metric to plot. Options are:
                - "f1_score": Plots F1-score vs. confidence threshold.
                - "recall": Plots Recall vs. confidence threshold.
                - "precision": Plots Precision vs. confidence threshold.
                - "all": Plots F1-score, Recall, and Precision together.
                Defaults to "all".

        Raises:
            AssertionError: If an invalid metric_type is provided.

        Displays:
            A line plot of the selected metric(s) against confidence thresholds.
        """

        assert metric_type in [
            "f1_score",
            "recall",
            "precision",
            "all",
        ], f"Provide the metric_type among ['f1_score', 'recall', 'precision', 'all']"

        df = self.overall_res_df
        plt.figure(figsize=(8, 5))

        if metric_type == "all":
            plt.plot(df["conf"], df["precision"], marker="o", label="Precision")
            plt.plot(df["conf"], df["recall"], marker="s", label="Recall")
            plt.plot(df["conf"], df["f1_score"], marker="^", label="F1 Score")
        else:
            plt.plot(
                df["conf"], df[metric_type], marker="o", label=metric_type.capitalize()
            )

        plt.xlabel("Confidence Threshold")
        plt.ylabel("Value")
        plt.title(
            f"{metric_type.capitalize()} vs Confidence Threshold"
            if metric_type != "all"
            else "Metrics vs Confidence Threshold"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_class_wise_result(self):
        """
        returns Dataframe of class wise stats on dataset
        """
        assert (
            self.class_vise_res_df is not None
        ), f"Plese use get_stats on the object first"
        return self.class_vise_res_df

    def get_dataset_result(self):
        """
        returns Dataframe of  stats  on complete dataset
        """
        assert (
            self.overall_res_df is not None
        ), f"Plese use get_stats on the object first"
        return self.overall_res_df

    def get_mAP_result(self):
        """
        returns Dataframe of mAP stats on dataset
        """
        assert self.mAP_df is not None, f"Plese use get_stats on the object first"
        return self.mAP_df

    def get_file_stats(self, filename):
        assert (
            filename in self.all_image_stats
        ), "File name not present, Please provide correct name"
        return self.all_image_stats[filename]
