import contextlib
import io
import json
import random
import threading
import traceback
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from core_assist.image_ops.src.image_utils import load_rgb
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
    def __init__(self, gt_json_path: str, pred_json_path: str):
        self.f1_score = {}
        self.precision = {}
        self.recall = {}
        self.miou = {}
        self.anno_path = gt_json_path
        self.img_path = None
        self.pred_path = pred_json_path
        self.results = {}
        self.height = None
        self.width = None

    def plot(
        self,
        bbox=True,
        original_image=False,
        ground_truth_masks=False,
        ground_truth=True,
        pred_masks=False,
        predictions=True,
        segment_type="both",
    ):
        """
        Visualizes the segmentation results by plotting the original image,
        ground truth segmentation, predicted segmentation, and their masks.

        Args:
            bbox (bool): If True, draws bounding boxes around the masks.
            original_image (bool): If True, displays the original image.
            ground_truth_mask (bool): If True, displays the ground truth mask.
            ground_truth (bool): If True, displays the segmented ground truth image.
            pred_mask (bool): If True, displays the predicted segmentation mask.
            predictions (bool): If True, displays the segmented prediction image.
            segment_type (optional) : Detrimines the type of segment drawn , possibel values ["both" , "outline" , "filled"]

        Raises:
            ValueError: If the image path is not set.

        Returns:
            None: Displays the selected images using `plot.image()`.
        """

        gt_label, predictions_data = self.load_json()

        self.img_path = gt_label["image_path"]
        self.height = gt_label["height"]
        self.width = gt_label["width"]

        if self.img_path is None:
            raise ValueError("Image path is not set.")

        img = load_rgb(self.img_path)

        # Extract ground truth masks and labels
        gt_masks = [
            mask_util.decode(item["segmentation"]) for item in gt_label["annotations"]
        ]
        gt_labels = [item["label"] for item in gt_label["annotations"]]

        # Extract predicted masks, labels, and confidence scores
        pred_mask = [
            mask_util.decode(item["segmentation"])
            for item in predictions_data["predictions"]
        ]
        pred_labels = [
            item["category_name"] for item in predictions_data["predictions"]
        ]
        pred_confs = [item["score"] for item in predictions_data["predictions"]]

        # Generate segmented images
        gt_img, gt_mask = plot.segment(
            img,
            gt_masks,
            gt_labels,
            ret=True,
            bbox_flag=bbox,
            segment_type=segment_type,
        )
        pred_img, pred_mask = plot.segment(
            img,
            pred_mask,
            pred_labels,
            confs=pred_confs,
            ret=True,
            bbox_flag=bbox,
            segment_type=segment_type,
        )

        # Collect images to display based on the boolean flags
        images_to_plot = {}

        if original_image:
            images_to_plot["original_image"] = img
        if ground_truth_masks:
            images_to_plot["ground_truth_mask"] = gt_mask
        if ground_truth:
            images_to_plot["ground_truth"] = gt_img
        if pred_masks:
            images_to_plot["pred_masks"] = pred_mask
        if predictions:
            images_to_plot["predictions"] = pred_img

        # Display only the selected images
        if images_to_plot:
            plot.image(**images_to_plot)

    def check_ann(self):
        pass

    def get_stats(
        self,
        iou_thresh=0.5,
        conf_thresh=[0.5],
        mode="conf",
        level="instance",
        tabulate=False,
    ):
        """
        Evaluates object segmentation performance for a single image by computing statistics such as
        precision, recall, F1-score, and IoU at different confidence thresholds.

        Args:
            gt_json_path (str): path to ground truth json path
            pred_json_path (str): path to prediction json path.
            iou_thresh (float, optional): IoU threshold for determining True Positives. Default is 0.3.
            conf_thresh (list, optional): List of confidence score thresholds for filtering predictions. Default is [0.5].
            mode (str, optional): Evaluation mode. "conf" for confidence-based filtering (default).
            level (str, optional): Level of evaluation: "instance" or "pixel". Default is "instance".
            tabulate (bool, optional): If True, return the stats in a tabulated form.

        Returns:
            dict: A dictionary containing performance metrics including F1-score, precision, recall, mIoU.
            class wise stats
        """

        if mode == "conf":
            assert len(conf_thresh) > 0, "Please provide confidence thresholds"
        if mode == "iou":
            assert len(iou_thresh) > 0, "Please provide IoU thresholds"

        assert level in [
            "instance",
            "pixel",
        ], "Level must be either 'instance' or 'pixel'"

        gt_label, predictions = self.load_json()

        self.filename = Path(gt_label["image_path"]).name
        self.img_path = gt_label["image_path"]

        # self.img_path = gt_label[0]["img_path"]
        # filename = Path(gt_label[0]["img_path"]).name

        self.height = gt_label["height"]
        self.width = gt_label["width"]

        gt_classes = set(item["label"] for item in gt_label["annotations"])
        # h, w = size
        results = {}

        for conf in conf_thresh:
            class_wise_stats = {}
            filtered_pred = [
                item for item in predictions["predictions"] if item["score"] >= conf
            ]
            # pred_classes = set([item['category_name'] for item in filtered_pred])
            pred_classes = set(item["category_name"] for item in filtered_pred)
            unique_classes = gt_classes.union(pred_classes)

            for class_name in unique_classes:
                # filtered_class_pred = [item for item in filtered_pred if item['category_name'] == class_name]
                filtered_class_pred = [
                    item
                    for item in filtered_pred
                    if item["category_name"] == class_name
                ]
                filtered_gt = [
                    item
                    for item in gt_label["annotations"]
                    if item["label"] == class_name
                ]

                filtered_gt_masks = [
                    mask_util.decode(item["segmentation"]) for item in filtered_gt
                ]

                filtered_pred_masks = [
                    mask_util.decode(item["segmentation"])
                    for item in filtered_class_pred
                ]

                if level == "instance":
                    tp, fp, fn, tn = self.check_iou(
                        filtered_gt_masks, filtered_pred_masks, iou_th=iou_thresh
                    )

                    iou = self.get_iou(
                        (
                            np.logical_or.reduce(filtered_gt_masks)
                            if filtered_gt
                            else np.zeros((self.height, self.width), dtype=np.uint8)
                        ),
                        (
                            np.logical_or.reduce(filtered_pred_masks)
                            if filtered_class_pred
                            else np.zeros((self.height, self.width), dtype=np.uint8)
                        ),
                    )

                    tp, fp, fn, tn, iou = self.check_pixel_iou(
                        (
                            np.logical_or.reduce(filtered_gt_masks)
                            if filtered_gt
                            else np.zeros((self.height, self.width), dtype=np.uint8)
                        ),
                        (
                            np.logical_or.reduce(filtered_pred_masks)
                            if filtered_class_pred
                            else np.zeros((self.height, self.width), dtype=np.uint8)
                        ),
                        iou_th=iou_thresh,
                    )

                class_wise_stats[class_name] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "iou": iou,
                    "gt": len(filtered_gt),
                }

            if class_wise_stats:
                f1, precision, recall, miou = self.calculate_image_level_metric(
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
            else:
                results[conf] = {
                    "filename": self.filename,
                    "f1": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "miou": 1.0,
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
        if self.anno_path and self.pred_path is None:
            raise ValueError("jsons paths are not set.")
        with open(self.anno_path, "r") as raw_gt:
            gt = json.load(raw_gt)
        with open(self.pred_path, "r") as raw_pred:
            pred = json.load(raw_pred)

        return gt, pred

    def get_iou(self, mask1, mask2):
        inter = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        if np.sum(union) == 0:
            return 0
        iou = np.sum(inter) / np.sum(union)
        return iou

    def check_pixel_iou(self, gt_mask, pred_mask, iou_th=0.3):
        """
        Compute TP, FP, FN, and TN at the pixel level for segmentation masks.

        :param gt_mask: Binary 2D numpy array (ground truth mask)
        :param pred_mask: Binary 2D numpy array (predicted mask)
        :param iou_th: IoU threshold for determining matches (default = 0.3)
        :return: TP, FP, FN, TN
        """
        iou = self.get_iou(gt_mask, pred_mask)

        tp = np.sum((pred_mask == 1) & (gt_mask == 1))
        fp = np.sum((pred_mask == 1) & (gt_mask == 0))
        tn = np.sum((pred_mask == 0) & (gt_mask == 0))
        fn = np.sum((pred_mask == 0) & (gt_mask == 1))

        return tp, fp, fn, tn, iou

    def check_iou(self, gt_masks, pred_masks, iou_th=0.3):
        tp, fp, fn, tn = 0, 0, 0, 0
        matches = {}
        for p_idx, p_mask in enumerate(pred_masks):
            for g_idx, g_mask in enumerate(gt_masks):
                iou = self.get_iou(p_mask, g_mask)
                if iou > iou_th:
                    if p_idx not in matches:
                        matches[p_idx] = [g_idx]
                    else:
                        matches[p_idx].append(g_idx)

        tp = len(matches)
        fp = len(pred_masks) - len(set(list(matches.keys())))
        fn = len(gt_masks) - len(
            set([item for sublist in list(matches.values()) for item in sublist])
        )
        return tp, fp, fn, tn

    def calculate_image_level_metric(self, class_wise_stats):
        tp = sum([item["tp"] for item in class_wise_stats.values()])
        fp = sum([item["fp"] for item in class_wise_stats.values()])
        fn = sum([item["fn"] for item in class_wise_stats.values()])

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        miou = np.mean([item["iou"] for item in class_wise_stats.values()])
        return f1, precision, recall, miou


import time


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
        start_time = time.time()  # Start time measurement

        def process_file(gt_path, pred_path):
            """
            Process a single file pair and compute statistics.
            """
            img_obj = Image(gt_path, pred_path).get_stats(
                conf_thresh=conf_thresh, iou_thresh=iou_thresh
            )

            with lock:
                filename = Path(img_obj.img_path).name
                if filename in self.all_image_stats:
                    self.all_image_stats[filename].append(img_obj)
                else:
                    self.all_image_stats[filename] = [img_obj]

                all_class_stats.append(img_obj.results)

        # Use joblib for parallel processing with tqdm for progress tracking
        with tqdm(total=len(self.gt_file_paths), desc="Processing files") as pbar:
            Parallel(n_jobs=-1, backend="threading")(
                delayed(lambda x: (process_file(*x), pbar.update(1)))(
                    (gt_path, pred_path)
                )
                for gt_path, pred_path in zip(self.gt_file_paths, self.pred_file_paths)
            )

        # Convert dataset to COCO format for mAP calculation
        coco_gtr, label_mapping = transform_to_coco(
            self.gt_file_paths, task="segmentation", is_gt=True
        )
        coco_pred = transform_to_coco(
            self.pred_file_paths,
            task="segmentation",
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
            coco_eval = COCOeval(coco_gt, coco_preds, "segm")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        # Compute overall dataset statistics
        tabulated_mAP, mAP_df = tabulate_map_res(coco_eval=coco_eval)
        class_wise_stats, dataset_stats = calculate_overall_metrics(all_class_stats)
        print(generate_overall_dataset_tabulated_stats(class_wise_stats, dataset_stats))
        print(tabulated_mAP)

        # Print total elapsed time
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
            by (str): The metric to filter by (e.g., "f1_score", "precision", "recall", "miou").
            thresh (float): The threshold value for the metric.
            eq (str): The comparison operator ("<", "<=", ">", ">=", "==").
            conf_thresh (float, optional): Confidence threshold to filter on.

        Returns:
            DatasetStats: A new DatasetStats object containing only the filtered images.
        """
        if conf_thresh is None:
            if len(self.conf_thresh) == 1:
                conf_thresh = self.conf_thresh[0]
            else:
                raise ValueError(f"Provide conf_thresh among {self.conf_thresh}")
        else:
            if conf_thresh not in self.conf_thresh:
                raise ValueError(f"Provide conf_thresh among {self.conf_thresh}")

        if not self.all_image_stats:
            return self

        new_image_stats = {}

        for filename, img_list in self.all_image_stats.items():
            filtered_imgs = []
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
                    filtered_imgs.append(img_obj)

            if filtered_imgs:
                new_image_stats[filename] = filtered_imgs

        # Create a new DatasetStats object with the filtered images
        filtered_dataset = DatasetStats([], [])
        filtered_dataset.all_image_stats = new_image_stats
        filtered_dataset.class_vise_res_df = self.class_vise_res_df
        filtered_dataset.overall_res_df = self.overall_res_df
        filtered_dataset.mAP_df = self.mAP_df

        return filtered_dataset

    def plot(
        self,
        samples=3,
        bbox=True,
        segment_type="both",
        original_image=False,
        ground_truth_mask=False,
        ground_truth=True,
        pred_mask=False,
        predictions=True,
    ):
        """
        Visualizes ground truth and predicted bounding boxes for a subset of images.

        Args:
            samples (int): Number of images to plot. Defaults to 3.
            bbox (bool): If True, draws bounding boxes around the masks.
            original_image (bool): If True, displays the original image.
            ground_truth_mask (bool): If True, displays the ground truth mask.
            ground_truth (bool): If True, displays the segmented ground truth image.
            pred_mask (bool): If True, displays the predicted segmentation mask.
            predictions (bool): If True, displays the segmented prediction image.

        Raises:
            ValueError: If no image statistics are available for plotting.

        Returns:
            None: Displays the selected images using their respective `plot()` methods.
        """

        if not self.all_image_stats:
            print("No image statistics available to plot.")
            return

        # Flatten dictionary values into a single list of images
        all_images = [
            img for img_list in self.all_image_stats.values() for img in img_list
        ]

        # Select random images if samples < total available images
        selected_images = random.sample(all_images, min(samples, len(all_images)))

        # Plot the selected images
        for i, img_obj in enumerate(selected_images, start=1):
            print(f"Plotting image {i}: {img_obj.img_path}")
            img_obj.plot(
                bbox=bbox,
                segment_type=segment_type,
                original_image=original_image,
                ground_truth_masks=ground_truth_mask,
                ground_truth=ground_truth,
                pred_masks=pred_mask,
                predictions=predictions,
            )

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
            self.class_vise_res_df is None
        ), f"Plese use get_stats on the object first"
        return self.class_vise_res_df

    def get_dataset_result(self):
        """
        returns Dataframe of  stats  on complete dataset
        """
        assert self.overall_res_df is None, f"Plese use get_stats on the object first"
        return self.overall_res_df

    def get_mAP_result(self):
        """
        returns Dataframe of mAP stats on dataset
        """
        assert self.mAP_df is None, f"Plese use get_stats on the object first"
        return self.mAP_df
