"""
Computer Vision Utility Library - Dataset Base Classes

This module provides base classes for handling computer vision datasets in various formats.
It includes support for COCO-style annotations and segmentation masks.

__author__: HashTagML
license: MIT
Created: Monday, 29th March 2021
"""

import os
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from core_assist.dataset.format import FormatSpec
from core_assist.dataset.utils import (exists, get_annotation_dir,
                                       get_image_dir, read_coco)
from joblib import Parallel, delayed
from tqdm.auto import tqdm

NUM_THREADS = os.cpu_count() // 2
import json
from functools import partial
from pathlib import Path
from typing import Dict


class Base(FormatSpec):
    """Base class for handling COCO-style object detection annotations.

    This class provides functionality to load and process object detection datasets
    that follow the COCO annotation format. It supports both split (train/val/test)
    and non-split dataset organizations.

    Args:
        root (Union[str, os.PathLike]): Path to root directory. Expects the ``root`` directory to have either
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
                    ├── train
                    |   ├── 1.json
                    |   ├── 2.json
                    |   │   ...
                    |   └── n.json
                    ├── valid(...)
                    └── test(...)

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
                    ├── 1.json
                    ├── 2.json
                    │   ...
                    └── n.json

        format (Optional[str]): Format specification for the dataset. Defaults to None.
    """

    def __init__(self, root: Union[str, os.PathLike], format: Optional[str] = None):
        super().__init__(root, format=format)  # Pass `format` to parent class
        self._image_dir = get_image_dir(root)
        self._annotation_dir = get_annotation_dir(root)
        assert os.path.exists(self._image_dir), "root is missing `images` directory."
        assert os.path.exists(
            self._annotation_dir
        ), "root is missing `annotations` directory."

        self._find_splits()
        self._resolve_dataframe()

    def _get_class_map(self, categories: List):
        """Creates a mapping from category names to unique class IDs.

        Args:
            categories (List): List of category names

        Returns:
            Dict: Mapping from category name to integer class ID
        """
        class_map = {cat: idx for idx, cat in enumerate(sorted(set(categories)))}
        return class_map

    def _resolve_dataframe(self):
        """Processes all annotations into a unified pandas DataFrame.

        Creates a master DataFrame containing all annotations across splits with standardized
        column names and data types. Handles parallel processing of annotation files for improved
        performance.
        """
        master_df = pd.DataFrame(
            columns=[
                "split",
                "image_id",
                "image_width",
                "image_height",
                "x_min",
                "y_min",
                "width",
                "height",
                "category",
                "image_path",
            ],
        )

        print("Loading COCO annotations:")
        for split in self._splits:
            (
                image_ids,
                image_paths,
                class_ids,
                x_mins,
                y_mins,
                bbox_widths,
                bbox_heights,
                image_heights,
                image_widths,
            ) = ([], [], [], [], [], [], [], [], [])
            split = split if self._has_image_split else ""
            annotations = Path(self._annotation_dir).joinpath(split).glob("*.json")

            parse_partial = partial(self._parse_json_file, split)
            all_instances = Parallel(n_jobs=NUM_THREADS, backend="multiprocessing")(
                delayed(parse_partial)(json_file)
                for json_file in tqdm(annotations, desc=split)
            )

            for instances in all_instances:
                image_ids.extend(instances["image_ids"])
                image_paths.extend(instances["image_paths"])
                class_ids.extend(instances["class_ids"])
                x_mins.extend(instances["x_mins"])
                y_mins.extend(instances["y_mins"])
                bbox_widths.extend(instances["bbox_widths"])
                bbox_heights.extend(instances["bbox_heights"])
                image_widths.extend(instances["image_widths"])
                image_heights.extend(instances["image_heights"])

            annots_df = pd.DataFrame(
                list(
                    zip(
                        image_ids,
                        image_paths,
                        image_widths,
                        image_heights,
                        class_ids,
                        x_mins,
                        y_mins,
                        bbox_widths,
                        bbox_heights,
                    )
                ),
                columns=[
                    "image_id",
                    "image_path",
                    "image_width",
                    "image_height",
                    "class_id",
                    "x_min",
                    "y_min",
                    "width",
                    "height",
                ],
            )
            annots_df["split"] = split if split else "main"
            master_df = pd.concat([master_df, annots_df], ignore_index=True)

        categories = master_df["category"].unique().tolist()
        label_map = self._get_class_map(categories)
        master_df["class_id"] = master_df["category"].map(label_map)
        self.master_df = master_df

    def _parse_json_file(self, split: str, json_path: Union[str, os.PathLike]) -> Dict:
        """Parses a single JSON annotation file in COCO format.

        Args:
            split (str): Dataset split name (e.g., 'train', 'valid', 'test')
            json_path (Union[str, os.PathLike]): Path to JSON annotation file

        Returns:
            Dict: Dictionary containing parsed annotations with standardized keys
        """
        label_info_keys = [
            "image_ids",
            "image_paths",
            "class_ids",
            "x_mins",
            "y_mins",
            "bbox_widths",
            "bbox_heights",
            "image_heights",
            "image_widths",
        ]
        label_info = {key: [] for key in label_info_keys}

        with open(json_path, "r") as f:
            data = json.load(f)

        image_path = data.get("image_path", "")
        image_name = data.get("image_name", "")
        image_width = data.get("width", 0)
        image_height = data.get("height", 0)

        for annotation in data.get("annotations", []):
            label = annotation["label"]
            x_min, y_min, x_max, y_max = annotation["bbox"]
            width = x_max - x_min
            height = y_max - y_min

            label_info["image_ids"].append(image_name)
            label_info["image_paths"].append(image_path)
            label_info["class_ids"].append(label)
            label_info["x_mins"].append(x_min)
            label_info["y_mins"].append(y_min)
            label_info["bbox_widths"].append(width)
            label_info["bbox_heights"].append(height)
            label_info["image_widths"].append(image_width)
            label_info["image_heights"].append(image_height)

        return label_info


import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


class SegmentationBase(Base):
    """Extended base class for handling COCO-style segmentation annotations.

    This class extends the Base class to specifically handle segmentation masks in COCO RLE format.
    It maintains the same directory structure expectations as the Base class but processes
    segmentation annotations instead of bounding boxes.

    Args:
        root (Union[str, os.PathLike]): Path to dataset root directory
        format (Optional[str]): Format specification for the dataset. Defaults to None.
    """

    def __init__(self, root: Union[str, os.PathLike], format: Optional[str] = None):
        super().__init__(root, format=format)  # Pass `format` to parent class
        self._image_dir = get_image_dir(root)
        self._annotation_dir = get_annotation_dir(root)
        assert os.path.exists(self._image_dir), "root is missing `images` directory."
        assert os.path.exists(
            self._annotation_dir
        ), "root is missing `annotations` directory."

        self._find_splits()
        self._resolve_dataframe()

    def _resolve_dataframe(self):
        """Override of Base._resolve_dataframe to handle segmentation data.

        Creates a master DataFrame specifically structured for segmentation annotations,
        including RLE-encoded mask data.
        """
        master_df = pd.DataFrame(
            columns=[
                "split",
                "image_id",
                "image_width",
                "image_height",
                "category",
                "image_path",
                "segmentation",
            ],
        )

        print("Loading COCO segmentation annotations:")
        for split in self._splits:
            image_ids, image_paths, categories, segmentations = [], [], [], []
            image_heights, image_widths = [], []

            split = split if self._has_image_split else ""
            annotations = Path(self._annotation_dir).joinpath(split).glob("*.json")

            parse_partial = partial(self._parse_json_file, split)
            all_instances = Parallel(n_jobs=NUM_THREADS, backend="multiprocessing")(
                delayed(parse_partial)(json_file)
                for json_file in tqdm(annotations, desc=split)
            )

            for instances in all_instances:
                image_ids.extend(instances["image_ids"])
                image_paths.extend(instances["image_paths"])
                categories.extend(instances["categories"])
                segmentations.extend(instances["segmentations"])
                image_heights.extend(instances["image_heights"])
                image_widths.extend(instances["image_widths"])

            annots_df = pd.DataFrame(
                list(
                    zip(
                        image_ids,
                        image_paths,
                        image_widths,
                        image_heights,
                        categories,
                        segmentations,
                    )
                ),
                columns=[
                    "image_id",
                    "image_path",
                    "image_width",
                    "image_height",
                    "category",
                    "segmentation",
                ],
            )
            annots_df["split"] = split if split else "main"
            master_df = pd.concat([master_df, annots_df], ignore_index=True)

        categories = master_df["category"].unique().tolist()
        label_map = self._get_class_map(categories)
        master_df["class_id"] = master_df["category"].map(label_map)
        self.master_df = master_df

    def _parse_json_file(self, split: str, json_path: Union[str, os.PathLike]) -> Dict:
        """Parses a single JSON annotation file containing segmentation data.

        Args:
            split (str): Dataset split name (e.g., 'train', 'valid', 'test')
            json_path (Union[str, os.PathLike]): Path to JSON annotation file

        Returns:
            Dict: Dictionary containing parsed segmentation annotations with standardized keys
        """
        label_info = {
            "image_ids": [],
            "image_paths": [],
            "categories": [],
            "segmentations": [],
            "image_heights": [],
            "image_widths": [],
        }

        with open(json_path, "r") as f:
            data = json.load(f)

        image_path = data.get("image_path", "")
        image_name = data.get("image_name", "")
        image_width = data.get("width", 0)
        image_height = data.get("height", 0)

        for annotation in data.get("annotations", []):
            label = annotation["label"]
            segmentation = annotation["segmentation"]

            label_info["image_ids"].append(image_name)
            label_info["image_paths"].append(image_path)
            label_info["categories"].append(label)
            label_info["segmentations"].append(segmentation)
            label_info["image_widths"].append(image_width)
            label_info["image_heights"].append(image_height)

        return label_info
