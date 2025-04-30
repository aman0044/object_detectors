import json
import traceback

import numpy as np
import pandas as pd


def load_json(json_path):
    """
    Loads ground truth or prediction data from a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        list: A list of dictionaries, each containing 'bbox', 'category_name', and 'score'.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(
            f"Error loading JSON file {json_path}: {traceback.format_exc()}"
        )

    labels = []

    if isinstance(data, list):
        # Handles predictions (list of dictionaries format)
        for item in data:
            if not all(key in item for key in ["bbox", "label", "score"]):
                raise ValueError(
                    f"Invalid prediction format in {json_path}. Missing required keys."
                )
            labels.append(
                {
                    "bbox": item["bbox"],
                    "score": item["score"],
                    "category_name": item["label"],
                }
            )
    elif isinstance(data, dict) and "annotations" in data:
        # Handles ground truth (COCO-style format with 'annotations' key)
        for item in data["annotations"]:
            if not all(key in item for key in ["bbox", "label"]):
                raise ValueError(
                    f"Invalid ground truth format in {json_path}. Missing required keys."
                )
            labels.append(
                {
                    "img_path": data["image_path"],
                    "bbox": item["bbox"],
                    "category_name": item["label"],
                    "score": 1.0,  # Default score for ground truth
                }
            )
    else:
        raise ValueError(f"Unsupported JSON format in {json_path}.")

    return labels


from tabulate import tabulate


def generate_classwise_tabulated_stats(image_stats):
    """
    Generate two separate tables: general statistics and class-wise statistics.

    :param image_stats: Dictionary containing statistics for different thresholds.
    :return: Formatted tabulated strings for both tables.
    """
    general_table = []
    classwise_table = []

    for threshold, data in image_stats.items():
        # General statistics
        general_table.append(
            [
                threshold,
                data["filename"],
                data["f1"],
                data["precision"],
                data["recall"],
                data["miou"],
            ]
        )

        # Class-wise statistics header
        classwise_table.append(
            ["Threshold", "Class Name", "TP", "FP", "FN", "IoU", "GT"]
        )

        for class_name, values in data["class_stats"].items():
            classwise_table.append(
                [
                    threshold,
                    class_name,
                    values["tp"],
                    values["fp"],
                    values["fn"],
                    values["iou"],
                    values["gt"],
                ]
            )

        classwise_table.append(["-" * 10] * 7)  # Separator

    # Formatting tables
    general_table_str = tabulate(
        general_table,
        headers=[
            "Threshold",
            "Filename",
            "F1 Score",
            "Precision",
            "Recall",
            "Mean IoU",
        ],
        tablefmt="pretty",
    )

    classwise_table_str = tabulate(
        classwise_table, headers="firstrow", tablefmt="pretty"
    )

    return f"General Statistics:\n{general_table_str}\n\nClass-wise Statistics:\n{classwise_table_str}"


def generate_overall_dataset_tabulated_stats(class_wise_stats, dataset_stats):
    """
    Generate a tabulated representation of class-level and overall dataset statistics.

    Args:
        class_level_metrics (dict): Class-level metrics per confidence threshold.
        overall_results (dict): Overall dataset metrics per confidence threshold.

    Returns:
        str: Tabulated statistics for class-level and overall metrics.
    """
    # Preparing class-level stats
    class_table = []
    for conf, class_stats in class_wise_stats.items():
        for class_name, stats in class_stats.items():
            class_table.append(
                [
                    conf,
                    class_name,
                    stats["precision"],
                    stats["recall"],
                    stats["f1_score"],
                    stats["miou"],
                    stats["total_gt_inst"],
                    stats["total_pred_inst"],
                ]
            )

    class_headers = [
        "Confidence Threshold",
        "Class Name",
        "Precision",
        "Recall",
        "F1 Score",
        "mIoU",
        "Total GT Inst",
        "Total Pred Inst",
    ]
    class_tabulated = tabulate(class_table, headers=class_headers, tablefmt="grid")

    # Preparing overall stats
    overall_table = []
    for conf, stats in dataset_stats.items():
        overall_table.append(
            [
                conf,
                stats["precision"],
                stats["recall"],
                stats["f1_score"],
                stats["miou"],
                stats["total_gt_inst"],
                stats["total_pred_inst"],
            ]
        )

    overall_headers = [
        "Confidence Threshold",
        "Precision",
        "Recall",
        "F1 Score",
        "mIoU",
        "Total GT Inst",
        "Total Pred Inst",
    ]
    overall_tabulated = tabulate(
        overall_table, headers=overall_headers, tablefmt="grid"
    )

    return f"Class-Level Metrics:\n{class_tabulated}\n\nOverall Metrics:\n{overall_tabulated}"


def class_metric_to_df(result):
    """
    Convert class-level metrics dictionary into a DataFrame.

    Args:
        result (dict): A dictionary containing class-level metrics for different confidence thresholds.

    Returns:
        pd.DataFrame: A DataFrame with columns ['conf', 'metric', 'class1', 'class2', ...].
    """
    rows = []
    for conf, metrics in result.items():
        classes = list(metrics.keys())
        for metric in metrics[classes[0]].keys():
            # Create a row for each metric
            row = {"conf": conf, "metric": metric}
            for cls in classes:
                row[cls] = round(
                    metrics[cls][metric], ndigits=4
                )  # Truncate float values
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def overall_metric_to_df(overall_metrics):
    """
    Convert overall metrics dictionary into a DataFrame.

    Args:
        overall_metrics (dict): A dictionary containing overall metrics for different confidence thresholds.

    Returns:
        pd.DataFrame: A DataFrame with columns ['conf', 'precision', 'recall', 'f1_score', 'miou'].
    """
    rows = []
    for conf, metrics in overall_metrics.items():
        row = {
            "conf": conf,
            "precision": round(
                metrics["precision"], ndigits=5
            ),  # Truncate float values
            "recall": round(metrics["recall"], ndigits=5),
            "f1_score": round(metrics["f1_score"], ndigits=5),
            "miou": round(metrics["miou"], ndigits=5),
            "total_gt_inst": metrics["total_gt_inst"],
            "total_pred_inst": metrics["total_pred_inst"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def calculate_overall_metrics(all_class_stats):
    """
    Compute overall precision, recall, F1-score, and mAP across multiple files.

    Args:
        all_class_stats (list): List of class stats dictionaries from multiple files.

    Returns:
        tuple: A tuple containing class-level metrics dataframe and overall metrics dataframe.
    """
    overall_results = {}
    class_level_metrics = {}

    confidence_thresholds = set()
    for file_stats in all_class_stats:
        confidence_thresholds.update(file_stats.keys())

    for conf in sorted(confidence_thresholds):
        total_tp, total_fp, total_fn, total_gt = 0, 0, 0, 0
        iou_list = []
        class_level_metrics[conf] = {}

        for file_stats in all_class_stats:
            if conf in file_stats:
                iou_list.append(file_stats[conf]["miou"])
                class_stats = file_stats[conf]["class_stats"]

                for class_name, stats in class_stats.items():
                    total_tp += stats["tp"]
                    total_fp += stats["fp"]
                    total_fn += stats["fn"]
                    total_gt += stats["gt"]

                    if class_name not in class_level_metrics[conf]:
                        class_level_metrics[conf][class_name] = {
                            "tp": 0,
                            "fp": 0,
                            "fn": 0,
                            "gt": 0,
                            "miou": [],
                        }

                    class_level_metrics[conf][class_name]["tp"] += stats["tp"]
                    class_level_metrics[conf][class_name]["fp"] += stats["fp"]
                    class_level_metrics[conf][class_name]["fn"] += stats["fn"]
                    class_level_metrics[conf][class_name]["gt"] += stats["gt"]
                    class_level_metrics[conf][class_name]["miou"].append(stats["iou"])

        # Calculate class-level metrics
        for class_name, stats in class_level_metrics[conf].items():
            precision = stats["tp"] / (stats["tp"] + stats["fp"] + 1e-8)
            recall = stats["tp"] / (stats["tp"] + stats["fn"] + 1e-8)
            f1_score = 2 * precision * recall / (precision + recall + 1e-8)
            total_gt_inst = int(stats["gt"])
            total_pred_inst = int(stats["tp"] + stats["fp"])
            mean_iou = np.mean(stats["miou"]) if stats["miou"] else 0

            class_level_metrics[conf][class_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "miou": mean_iou,
                "total_gt_inst": total_gt_inst,
                "total_pred_inst": total_pred_inst,
            }

        # Calculate overall metrics (macro-averaged)
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        total_gt_inst = int(total_gt)
        total_pred_inst = int(total_tp + total_fp)
        mean_iou = np.mean(iou_list) if iou_list else 0

        overall_results[conf] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "miou": mean_iou,
            "total_gt_inst": total_gt_inst,
            "total_pred_inst": total_pred_inst,
        }

    return class_level_metrics, overall_results


import pandas as pd
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate


def tabulate_map_res(coco_eval: COCOeval):
    """
    Returns the COCO evaluation results in two formats:
    1. A tabulated string for printing.
    2. A Pandas DataFrame.

    Args:
        coco_eval (COCOeval): A COCOeval object after calling evaluate(), accumulate(), and summarize().

    Returns:
        tabulated_result (str): A formatted table for printing.
        result_df (pd.DataFrame): A DataFrame containing the evaluation results.

    Raises:
        ValueError: If the length of `coco_eval.stats` is not 12.
    """
    # Extract results from COCOeval
    results = coco_eval.stats  # This is a numpy array of 12 values

    # Define the metrics corresponding to the 12 values in `results`
    metrics = [
        "AP @ IoU=0.50:0.95 (all)",
        "AP @ IoU=0.50 (all)",
        "AP @ IoU=0.75 (all)",
        "AP @ IoU=0.50:0.95 (small)",
        "AP @ IoU=0.50:0.95 (medium)",
        "AP @ IoU=0.50:0.95 (large)",
        "AR @ IoU=0.50:0.95 (all)",
        "AR @ IoU=0.50 (all)",
        "AR @ IoU=0.75 (all)",
        "AR @ IoU=0.50:0.95 (small)",
        "AR @ IoU=0.50:0.95 (medium)",
        "AR @ IoU=0.50:0.95 (large)",
    ]

    # Check if the lengths match
    if len(results) != len(metrics):
        raise ValueError(
            f"Length mismatch: `coco_eval.stats` has {len(results)} values, "
            f"but `metrics` has {len(metrics)} entries. "
            "Ensure `coco_eval` is properly evaluated."
        )

    # Create a dictionary for the results
    result_dict = {"Metric": metrics, "Value": results}

    # Convert to a Pandas DataFrame
    result_df = pd.DataFrame(result_dict)

    # Create a tabulated string for printing
    tabulated_result = tabulate(
        result_dict, headers="keys", tablefmt="pretty", floatfmt=".3f"
    )

    return f"Overall mAP Metrics:\n{tabulated_result}", result_df


import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pycocotools.mask as mask_util


def transform_to_coco(
    json_file_paths, task="detection", is_gt=True, label_mapping=None, exclude_class=[]
):
    """
    Transform a list of custom JSON file paths into COCO format for either detection or segmentation.

    Args:
        json_file_paths (list): List of paths to JSON files.
        task (str): Task type, either "detection" or "segmentation".
        is_gt (bool): Flag to indicate if the data is ground truth (GT) or predictions.
        label_mapping (dict): Mapping from class names to category IDs for predictions.

    Returns:
        dict: COCO-formatted dictionary.
    """
    if task not in ["detection", "segmentation"]:
        raise ValueError("Task must be either 'detection' or 'segmentation'.")

    if is_gt:
        coco_data = {"images": [], "annotations": [], "categories": []}
    else:
        coco_data = []

    exclude_class = set(exclude_class)
    class_name_to_id = {}
    category_id = 1
    annotation_id = 1

    # Lock for thread-safe updates to shared variables
    lock = threading.Lock()

    def process_file(json_file_path, image_id):
        nonlocal category_id, annotation_id, exclude_class

        with open(json_file_path, "r") as file:
            json_data = json.load(file)

        image_data = []
        annotations_data = []

        if is_gt:
            image_path = json_data["image_path"]
            height, width = json_data["height"], json_data["width"]

            image_data.append(
                {
                    "id": image_id,
                    "file_name": image_path.rsplit("/", 1)[-1],
                    "height": height,
                    "width": width,
                }
            )

            for inst in json_data["annotations"]:
                # for inst in json_data["instances"] if task == "segmentation" else json_data["annotations"]:
                label = inst["label"]
                if label in exclude_class:
                    continue
                if task == "segmentation":
                    segmentation = inst["segmentation"]
                    area = mask_util.area(segmentation)
                else:
                    bbox = inst["bbox"]
                    if isinstance(bbox, np.ndarray):  # Convert NumPy array to list
                        bbox = bbox.tolist()

                with lock:  # Thread-safe updates
                    if label not in class_name_to_id:
                        class_name_to_id[label] = category_id

                        coco_data["categories"].append(
                            {
                                "id": category_id,
                                "name": label,
                            }
                        )
                        category_id += 1

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_name_to_id[label],
                    "iscrowd": 0,
                }

                if task == "segmentation":
                    annotation["segmentation"] = segmentation
                    annotation["area"] = float(area)
                else:
                    annotation["bbox"] = bbox
                    annotation["area"] = bbox[2] * bbox[3]

                annotations_data.append(annotation)
                annotation_id += 1
        else:
            for inst in json_data["predictions"]:
                # for inst in json_data if task == "segmentation" else json_data["predictions"]:
                label = (
                    inst["category_name"] if task == "segmentation" else inst["label"]
                )
                if label not in label_mapping:
                    continue

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label_mapping[label],
                    "iscrowd": 0,
                }

                if task == "segmentation":
                    annotation["segmentation"] = inst["segmentation"]
                    annotation["score"] = inst["score"]
                else:
                    bbox = inst["bbox"]
                    if isinstance(bbox, np.ndarray):  # Convert NumPy array to list
                        bbox = bbox.tolist()
                    annotation["bbox"] = bbox
                    annotation["score"] = inst["score"]

                if is_gt:
                    annotations_data.append(annotation)
                else:
                    annotations_data.append(annotation)

                annotation_id += 1

        return image_data, annotations_data

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, json_file_path, idx)
            for idx, json_file_path in enumerate(json_file_paths)
        ]
        for future in as_completed(futures):
            image_data, annotations_data = future.result()
            if is_gt:
                coco_data["images"].extend(image_data)
                coco_data["annotations"].extend(annotations_data)
            else:
                coco_data.extend(annotations_data)

    if is_gt:
        return coco_data, class_name_to_id
    else:
        return coco_data
