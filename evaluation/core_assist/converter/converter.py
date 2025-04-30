import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm


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


def bdd_to_core_assist(
    json_path, image_dir, class_map: dict, output_path, height=720, width=1280
):
    """
    Convert BDD JSON format to Core Assist format into individual json files.

    Args:
        json_path (str): Path to the BDD JSON file.
        output_path (str): Path to save the converted Core Assist JSON files.

    """
    os.makedirs(output_path, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    for image_data in tqdm(data):
        image_name = image_data["name"]
        image_path = os.path.join(image_dir, image_name)

        # Create a new JSON file for each image
        core_assist_json = {
            "image_path": image_path,
            "height": height,
            "width": width,
            "annotations": [],
        }

        # Find annotations for the current image
        for annotation in image_data["labels"]:
            label = class_map.get(annotation["category"], None)
            if label is not None:
                bbox = annotation.get("box2d", [])
                if len(bbox):
                    core_assist_json["annotations"].append(
                        {
                            "bbox": [
                                int(bbox["x1"]),
                                int(bbox["y1"]),
                                int(bbox["x2"]),
                                int(bbox["y2"]),
                            ],
                            "label": label,
                        }
                    )

        # Save the Core Assist JSON file
        output_file_path = (
            f"{output_path}/{image_path.split('/')[-1].replace('.jpg', '.json')}"
        )
        with open(output_file_path, "w") as out_f:
            json.dump(core_assist_json, out_f, indent=4)


def convert_bdd_to_yolo(bdd_json_path, output_dir, class_map, image_size=(1280, 720)):
    """
    Convert BDD100K object detection JSON annotations to YOLO format.

    Parameters:
        bdd_json_path (str): Path to the BDD JSON annotation file.
        output_dir (str): Directory to save YOLO .txt annotation files.
        class_map (dict): Mapping from BDD class names to YOLO class IDs.
        image_size (tuple): (width, height) of the images, default is (1280, 720) for BDD100K.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(bdd_json_path, "r") as f:
        data = json.load(f)

    for item in data:
        image_name = os.path.splitext(item["name"])[0]
        txt_file = os.path.join(output_dir, image_name + ".txt")
        lines = []
        for label in item.get("labels", []):
            category = label.get("category")
            if category not in class_map:
                continue
            class_id = class_map[category]
            box2d = label.get("box2d")
            if not box2d:
                continue
            x1, y1 = box2d["x1"], box2d["y1"]
            x2, y2 = box2d["x2"], box2d["y2"]
            x_center = ((x1 + x2) / 2) / image_size[0]
            y_center = ((y1 + y2) / 2) / image_size[1]
            width = (x2 - x1) / image_size[0]
            height = (y2 - y1) / image_size[1]
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            lines.append(line)

        with open(txt_file, "w") as f:
            f.write("\n".join(lines))
