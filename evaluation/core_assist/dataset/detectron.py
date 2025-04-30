import json
import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from core_assist.dataset.format import SegFormatSpec
from core_assist.dataset.utils import get_annotation_dir, get_image_dir
from joblib import Parallel, delayed
from pycocotools import mask as maskUtils
from tqdm import tqdm


class SegmentationDetectron(SegFormatSpec):
    """Represents a Detectron2 segmentation annotation object.

    This class handles loading and processing of Detectron2 format segmentation annotations.
    It supports both directory-based and CSV-based data organization.

    Args:
        root (Union[str, os.PathLike]): Path to root directory or a CSV file.
            If it's a directory, it expects the root directory to have either
            of the following layouts:

            .. code-block:: bash

                root
                ├── images
                │   ├── train
                │   │   ├── 1.jpg
                │   │   ├── 2.jpg
                │   │   │   ...
                │   │   └── n.jpg
                │   ├── valid (...)
                │   └── test (...)
                │
                └── annotations
                    ├── train.json
                    ├── valid.json
                    └── test.json

            or,

            .. code-block:: bash

                root
                ├── images
                │   ├── 1.jpg
                │   ├── 2.jpg
                │   │   ...
                │   └── n.jpg
                │
                └── annotations
                    └── label.json

            If it's a CSV file, it expects the following columns:
                - img_path: Path to the image file.
                - anno_path: Path to the annotation file (Detectron2 JSON format).
                - splits (optional): Split name (e.g., train, valid, test).

        format (Optional[str]): Format specification for the annotations.
        mapping (Optional[dict]): A dictionary mapping class IDs to class names.
            If not provided, class IDs will be used as string labels.

    Attributes:
        master_df (pd.DataFrame): A DataFrame containing all processed annotations with columns:
            - image_id: Unique identifier for each image
            - image_width: Width of the image in pixels
            - image_height: Height of the image in pixels
            - segmentation: RLE encoded segmentation mask
            - category: Class name or ID
            - class_id: Integer class identifier
            - image_path: Path to the image file
            - split: Dataset split (train/valid/test)
            - x_min: Bounding box left coordinate
            - y_min: Bounding box top coordinate
            - width: Bounding box width
            - height: Bounding box height
            - area: Area of the segmentation mask in pixels
    """

    def __init__(
        self, root: Union[str, os.PathLike], format: Optional[str] = None, mapping=None
    ):
        super().__init__(root, format=format)
        self._is_csv = False
        self.mapping = mapping

        # Check if root is a CSV file
        if isinstance(root, (str, os.PathLike)) and str(root).endswith(".csv"):
            self._is_csv = True
            self._csv_data = pd.read_csv(root)
            assert (
                "img_path" in self._csv_data.columns
            ), "CSV must have 'img_path' column."
            assert (
                "anno_path" in self._csv_data.columns
            ), "CSV must have 'anno_path' column."
            self._has_splits = "splits" in self._csv_data.columns
        else:
            # Handle directory case
            self._image_dir = get_image_dir(root)
            self._annotation_dir = get_annotation_dir(root)
            self._has_image_split = False
            assert (
                self._image_dir.exists()
            ), "Root directory is missing images directory."
            assert (
                self._annotation_dir.exists()
            ), "Root directory is missing annotations directory."
            self._find_splits()

        self._resolve_dataframe()

    def _resolve_dataframe(self):
        """Resolves the Detectron2 annotations into a master DataFrame.

        This method processes the annotation files and builds a unified DataFrame containing
        all segmentation information. It handles both CSV-based and directory-based data sources.

        For CSV sources, it processes annotations by splits if available. For directory sources,
        it processes each split's JSON file separately. The method converts polygon segmentations
        to RLE format and computes areas and bounding boxes.

        The resulting DataFrame is stored in self.master_df with standardized column names
        and data types. Class IDs are mapped to category names if a mapping was provided.
        """
        master_df = pd.DataFrame(
            columns=[
                "image_id",
                "image_width",
                "image_height",
                "segmentation",
                "category",
                "class_id",
                "image_path",
                "split",
                "x_min",
                "y_min",
                "width",
                "height",
                "area",
            ],
        )

        if self._is_csv:
            label_map = self.mapping
            if self._has_splits:
                # Group by splits and process each split separately
                grouped = self._csv_data.groupby("splits")
            else:
                # Treat all rows as one group if no splits are provided
                grouped = [(None, self._csv_data)]

            # for split, group in grouped:
            for split, group in tqdm(grouped, desc="Processing CSV splits"):
                # Read the first anno_path in the group
                anno_path = Path(group["anno_path"].iloc[0])
                detectron_data = self._read_detectron_json(anno_path)

                # Create a mapping from file_name to img_path for the current group
                image_path_map = dict(
                    zip(
                        group["img_path"].apply(lambda x: Path(x).name),
                        group["img_path"],
                    )
                )

                # Process each image in the group
                for img_data in detectron_data:
                    file_name = Path(img_data["file_name"]).name
                    if file_name in image_path_map:
                        img_path = image_path_map[file_name]
                        image_id = img_data["image_id"]
                        image_width = int(img_data["width"])
                        image_height = int(img_data["height"])

                        if "annotations" in img_data:
                            for ann in img_data["annotations"]:
                                segmentation = ann["segmentation"]
                                bbox = ann["bbox"]
                                class_id = int(ann["category_id"])

                                # Convert polygon segmentation to RLE
                                rle = self._polygon_to_rle(
                                    segmentation, image_height, image_width
                                )
                                if rle == {}:
                                    area = 0
                                else:
                                    area = maskUtils.area(rle).item()

                                # Add to DataFrame
                                master_df = pd.concat(
                                    [
                                        master_df,
                                        pd.DataFrame(
                                            [
                                                {
                                                    "image_id": image_id,
                                                    "image_width": image_width,
                                                    "image_height": image_height,
                                                    "segmentation": rle,
                                                    "category": f"class_{class_id}",
                                                    "class_id": class_id,
                                                    "image_path": str(img_path),
                                                    "split": (
                                                        split
                                                        if split is not None
                                                        else ""
                                                    ),
                                                    "x_min": bbox[0],
                                                    "y_min": bbox[1],
                                                    "width": bbox[2],
                                                    "height": bbox[3],
                                                    "area": area,
                                                }
                                            ]
                                        ),
                                    ],
                                    ignore_index=True,
                                )
        else:
            # Handle directory case (unchanged)
            # for split in self._splits:
            for split in tqdm(self._splits, desc="Processing directory splits"):
                coco_json = self._annotation_dir / f"{split}.json"
                detectron_data = self._read_detectron_json(coco_json)

                for img_data in detectron_data:
                    file_name = img_data["file_name"]
                    image_id = img_data["image_id"]
                    image_width = int(img_data["width"])
                    image_height = int(img_data["height"])

                    if "annotations" in img_data:
                        for ann in img_data["annotations"]:
                            segmentation = ann["segmentation"]
                            bbox = ann["bbox"]
                            class_id = int(ann["category_id"])

                            # Convert polygon segmentation to RLE
                            rle = self._polygon_to_rle(
                                segmentation, image_height, image_width
                            )
                            if rle == {}:
                                area = 0
                            else:
                                area = maskUtils.area(rle).item()

                            # Add to DataFrame
                            master_df = pd.concat(
                                [
                                    master_df,
                                    pd.DataFrame(
                                        [
                                            {
                                                "image_id": image_id,
                                                "image_width": image_width,
                                                "image_height": image_height,
                                                "segmentation": rle,
                                                "category": f"class_{class_id}",
                                                "class_id": class_id,
                                                "image_path": str(
                                                    self._image_dir / file_name
                                                ),
                                                "split": split,
                                                "x_min": bbox[0],
                                                "y_min": bbox[1],
                                                "width": bbox[2],
                                                "height": bbox[3],
                                                "area": area,
                                            }
                                        ]
                                    ),
                                ],
                                ignore_index=True,
                            )

        # Clean and finalize master DataFrame
        self.master_df = master_df
        master_df["class_id"] = master_df["class_id"].astype(np.int32)
        if self.mapping:
            master_df["category"] = master_df["class_id"].map(self.mapping)
        else:
            master_df["category"] = master_df["class_id"].astype(str)

    def _read_detectron_json(self, anno_path):
        """Reads and returns the Detectron2 JSON data.

        Args:
            anno_path (Union[str, Path]): Path to the JSON annotation file

        Returns:
            dict: The loaded JSON data containing image and annotation information
        """
        with open(anno_path, "r") as f:
            return json.load(f)

    def _polygon_to_rle(self, segmentation, height, width):
        """Converts polygon segmentation to RLE format.

        Args:
            segmentation (List[List[float]]): List of polygon vertices as [x1,y1,x2,y2,...]
            height (int): Height of the image
            width (int): Width of the image

        Returns:
            dict: RLE encoded mask with 'counts' and 'size' fields
        """
        if isinstance(segmentation, list) and len(segmentation) > 0:
            # Convert polygon to binary mask
            mask = np.zeros((height, width), dtype=np.uint8)
            for polygon in segmentation:
                polygon = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [polygon], 1)

            # Convert mask to RLE
            rle = maskUtils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle
        return {}

    def compute_area(self, segmentation):
        """Computes the area of a polygon-based segmentation.

        Args:
            segmentation (List[List[float]]): List of polygon vertices as [x1,y1,x2,y2,...]

        Returns:
            float: Area of the polygon in pixels
        """
        if isinstance(segmentation, list) and len(segmentation) > 0:
            # Use the first polygon to compute area
            polygon = np.array(segmentation[0]).reshape(-1, 2)
            return cv2.contourArea(polygon.astype(np.int32))
        return 0
