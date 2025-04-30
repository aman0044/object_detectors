import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from core_assist.dataset.format import FormatSpec, SegFormatSpec
from core_assist.dataset.utils import (exists, get_annotation_dir,
                                       get_image_dir, read_coco)
from pycocotools import mask as maskUtils
from tqdm import tqdm


class Coco(FormatSpec):
    """Represents a COCO annotation object.

    Args:
        root (Union[str, os.PathLike]): path to root directory or a CSV file.
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
                - `img_path`: Path to the image file.
                - `anno_path`: Path to the annotation file (COCO JSON format).
                - `splits` (optional): Split name (e.g., train, valid, test).
    """

    def __init__(
        self, root: Union[str, os.PathLike], format: Optional[str] = None, mapping=None
    ):
        """Initialize Coco object.

        Args:
            root: Path to root directory or CSV file
            format: Optional format specification
            mapping: Optional class mapping
        """
        self.root = root
        super().__init__(root, format=format)
        self.mapping = mapping
        self._is_csv = False

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
            assert exists(self._image_dir), "root is missing images directory."
            assert exists(
                self._annotation_dir
            ), "root is missing annotations directory."
            self._find_splits()

        self._resolve_dataframe()

    def _get_class_map(self, categories: List):
        """Map from category id to category name.

        Args:
            categories: List of category dictionaries

        Returns:
            Dictionary mapping category IDs to names
        """
        class_map = dict()
        for cat in categories:
            class_map[cat["id"]] = cat["name"]
        return class_map

    def _resolve_dataframe(self):
        """Resolve and create the master dataframe containing all annotations.

        Processes either CSV input or directory structure to create a unified
        dataframe with all image and annotation information.
        """
        split_str = []
        master_df = pd.DataFrame(
            columns=[
                "image_id",
                "image_width",
                "image_height",
                "x_min",
                "y_min",
                "width",
                "height",
                "category",
            ],
        )

        if self._is_csv:
            # Handle CSV case
            if self._has_splits:
                # Group by splits if splits are provided
                grouped = self._csv_data.groupby("splits")
            else:
                # Treat all rows as one group if no splits are provided
                grouped = [(None, self._csv_data)]

            # for split, group in grouped:
            for split, group in tqdm(grouped, desc="Processing CSV splits"):
                # Read the COCO JSON file for the current split (or once if no splits)
                anno_path = Path(
                    group["anno_path"].iloc[0]
                )  # Read the first annotation path in the group
                images, annots, cats = read_coco(anno_path)
                class_map = self._get_class_map(cats)

                # Create DataFrame for images
                images_df = pd.DataFrame(images)
                images_df = images_df[["id", "file_name", "width", "height"]]
                images_df.rename(
                    columns={"width": "image_width", "height": "image_height"},
                    inplace=True,
                )

                # Create DataFrame for annotations
                instances = [
                    (x["image_id"], x["category_id"], x["bbox"]) for x in annots
                ]
                annots_df = pd.DataFrame(
                    instances, columns=["image_id", "class_id", "bbox"]
                )
                annots_df["category"] = annots_df["class_id"].map(class_map)
                annots_df[["x_min", "y_min", "width", "height"]] = pd.DataFrame(
                    annots_df["bbox"].to_list(), index=annots_df.index
                )
                annots_df.drop(["bbox"], axis=1, inplace=True)

                # Merge annotations and images
                annots_df = annots_df.merge(
                    images_df, left_on="image_id", right_on="id", how="left"
                )
                annots_df.drop(["id", "image_id"], axis=1, inplace=True)
                annots_df.rename(columns={"file_name": "image_id"}, inplace=True)

                # Handle missing images
                null_images = annots_df["image_id"].isnull().sum()
                if null_images > 0:
                    warnings.warn(
                        "Some annotations in the dataset do not have images attached to them. Ignoring those annotations."
                    )
                annots_df.dropna(subset=["image_id"], inplace=True)

                # Map image paths from CSV to the COCO annotations
                image_path_map = dict(
                    zip(
                        group["img_path"].apply(lambda x: Path(x).name),
                        group["img_path"],
                    )
                )
                annots_df["image_path"] = annots_df["image_id"].map(image_path_map)

                # Add split (if available)
                annots_df["split"] = split if split is not None else ""

                # Append to master DataFrame
                master_df = pd.concat([master_df, annots_df], ignore_index=True)
        else:
            # Handle directory case (unchanged)
            # for split in self._splits:
            for split in tqdm(self._splits, desc="Processing directory splits"):
                coco_json = self._annotation_dir / f"{split}.json"
                images, annots, cats = read_coco(coco_json)
                split_str.append([split, len(images), len(annots), len(cats)])

                class_map = self._get_class_map(cats)

                images_df = pd.DataFrame(images)
                images_df = images_df[["id", "file_name", "width", "height"]]
                images_df.rename(
                    columns={"width": "image_width", "height": "image_height"},
                    inplace=True,
                )

                instances = [
                    (x["image_id"], x["category_id"], x["bbox"]) for x in annots
                ]
                annots_df = pd.DataFrame(
                    instances, columns=["image_id", "class_id", "bbox"]
                )
                annots_df["category"] = annots_df["class_id"].map(class_map)
                annots_df[["x_min", "y_min", "width", "height"]] = pd.DataFrame(
                    annots_df["bbox"].to_list(), index=annots_df.index
                )
                annots_df.drop(["bbox"], axis=1, inplace=True)

                annots_df = annots_df.merge(
                    images_df, left_on="image_id", right_on="id", how="left"
                )
                annots_df.drop(["id", "image_id"], axis=1, inplace=True)
                annots_df.rename(columns={"file_name": "image_id"}, inplace=True)

                null_images = annots_df["image_id"].isnull().sum()
                if null_images > 0:
                    warnings.warn(
                        "Some annotations in the dataset do not have images attached to them. Ignoring those annotations."
                    )
                annots_df.dropna(subset=["image_id"], inplace=True)
                annots_df["split"] = split
                split_dir = split if self._has_image_split else ""
                annots_df["image_path"] = annots_df["image_id"].map(
                    lambda x: self.root.joinpath("images")
                    .joinpath(split_dir)
                    .joinpath(x)
                )

                if len(annots_df[pd.isnull(annots_df.image_id)]) > 0:
                    warnings.warn(
                        "There are annotations in your dataset for which there is no matching images"
                        + f"(in split {split}). These annotations will be removed during any "
                        + "computation or conversion. It is recommended that you clean your dataset."
                    )

                master_df = pd.concat([master_df, annots_df], ignore_index=True)

        # Clean and finalize master DataFrame
        master_df = master_df[pd.notnull(master_df.image_id)]
        for col in ["x_min", "y_min", "width", "height"]:
            master_df[col] = master_df[col].astype(np.float32)

        for col in ["image_width", "image_height", "class_id"]:
            master_df[col] = master_df[col].astype(np.int32)

        self.master_df = master_df


import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class SegmentationCoco(SegFormatSpec):
    """Represents a COCO segmentation annotation object.

    Args:
        root (Union[str, os.PathLike]): path to root directory or a CSV file.
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
                - `img_path`: Path to the image file.
                - `anno_path`: Path to the annotation file (COCO JSON format).
                - `splits` (optional): Split name (e.g., train, valid, test).
    """

    def __init__(
        self, root: Union[str, os.PathLike], format: Optional[str] = None, mapping=None
    ):
        """Initialize SegmentationCoco object.

        Args:
            root: Path to root directory or CSV file
            format: Optional format specification
            mapping: Optional class mapping
        """
        super().__init__(root, format=format)
        self._is_csv = False

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
            ), "Root directory is missing `images` directory."
            assert (
                self._annotation_dir.exists()
            ), "Root directory is missing `annotations` directory."
            self._find_splits()

        self._resolve_dataframe()

    def _get_class_map(self, categories: List):
        """Map from category ID to category name.

        Args:
            categories: List of category dictionaries

        Returns:
            Dictionary mapping category IDs to names
        """
        return {cat["id"]: cat["name"] for cat in categories}

    def _resolve_dataframe(self):
        """Resolve and create the master dataframe containing all segmentation annotations.

        Processes either CSV input or directory structure to create a unified
        dataframe with all image and segmentation annotation information.
        """
        split_str = []
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
            # Handle CSV case
            if self._has_splits:
                # Group by splits if splits are provided
                grouped = self._csv_data.groupby("splits")
            else:
                # Treat all rows as one group if no splits are provided
                grouped = [(None, self._csv_data)]

            # for split, group in grouped:
            for split, group in tqdm(grouped, desc="Processing CSV splits"):
                # Read the COCO JSON file for the current split (or once if no splits)
                anno_path = Path(
                    group["anno_path"].iloc[0]
                )  # Read the first annotation path in the group
                images, annots, cats = read_coco(anno_path)
                class_map = self._get_class_map(cats)

                # Create DataFrame for images
                images_df = pd.DataFrame(images)
                images_df = images_df[["id", "file_name", "width", "height"]]
                images_df.rename(
                    columns={"width": "image_width", "height": "image_height"},
                    inplace=True,
                )

                # Create DataFrame for annotations
                instances = [
                    (x["image_id"], x["category_id"], x["segmentation"]) for x in annots
                ]
                annots_df = pd.DataFrame(
                    instances, columns=["image_id", "class_id", "segmentation"]
                )
                annots_df["category"] = annots_df["class_id"].map(class_map)

                # Compute bbox and area for each segmentation
                annots_df[["bbox", "area"]] = annots_df["segmentation"].apply(
                    lambda seg: pd.Series(self.compute_bbox_and_area(seg))
                )
                annots_df[["x_min", "y_min", "width", "height"]] = pd.DataFrame(
                    annots_df["bbox"].tolist(), index=annots_df.index
                )
                annots_df.drop("bbox", axis=1, inplace=True)

                # Merge annotations and images
                annots_df = annots_df.merge(
                    images_df, left_on="image_id", right_on="id", how="left"
                )
                annots_df.drop(["id", "image_id"], axis=1, inplace=True)
                annots_df.rename(columns={"file_name": "image_id"}, inplace=True)

                # Handle missing images
                null_images = annots_df["image_id"].isnull().sum()
                if null_images > 0:
                    warnings.warn(
                        "Some annotations in the dataset do not have images attached to them. Ignoring those annotations."
                    )
                annots_df.dropna(subset=["image_id"], inplace=True)

                # Map image paths from CSV to the COCO annotations
                image_path_map = dict(
                    zip(
                        group["img_path"].apply(lambda x: Path(x).name),
                        group["img_path"],
                    )
                )
                annots_df["image_path"] = annots_df["image_id"].map(image_path_map)

                # Add split (if available)
                annots_df["split"] = split if split is not None else ""

                # Append to master DataFrame
                master_df = pd.concat([master_df, annots_df], ignore_index=True)
        else:
            # Handle directory case (unchanged)
            # for split in self._splits:
            for split in tqdm(self._splits, desc="Processing directory splits"):
                coco_json = self._annotation_dir / f"{split}.json"
                images, annots, cats = read_coco(coco_json)
                split_str.append([split, len(images), len(annots), len(cats)])

                class_map = self._get_class_map(cats)

                images_df = pd.DataFrame(images)
                images_df = images_df[["id", "file_name", "width", "height"]]
                images_df.rename(
                    columns={"width": "image_width", "height": "image_height"},
                    inplace=True,
                )

                instances = [
                    (x["image_id"], x["category_id"], x["segmentation"]) for x in annots
                ]
                annots_df = pd.DataFrame(
                    instances, columns=["image_id", "class_id", "segmentation"]
                )
                annots_df["category"] = annots_df["class_id"].map(class_map)

                # Compute bbox and area for each segmentation
                annots_df[["bbox", "area"]] = annots_df["segmentation"].apply(
                    lambda seg: pd.Series(self.compute_bbox_and_area(seg))
                )
                annots_df[["x_min", "y_min", "width", "height"]] = pd.DataFrame(
                    annots_df["bbox"].tolist(), index=annots_df.index
                )
                annots_df.drop("bbox", axis=1, inplace=True)

                # Merge annotations and images
                annots_df = annots_df.merge(
                    images_df, left_on="image_id", right_on="id", how="left"
                )
                annots_df.drop(["id", "image_id"], axis=1, inplace=True)
                annots_df.rename(columns={"file_name": "image_id"}, inplace=True)

                # Handle missing images
                null_images = annots_df["image_id"].isnull().sum()
                if null_images > 0:
                    warnings.warn(
                        "Some annotations in the dataset do not have images attached to them. Ignoring those annotations."
                    )
                annots_df.dropna(subset=["image_id"], inplace=True)

                # Add split and image path
                annots_df["split"] = split
                split_dir = split if self._has_image_split else ""
                annots_df["image_path"] = annots_df["image_id"].map(
                    lambda x: self.root.joinpath("images")
                    .joinpath(split_dir)
                    .joinpath(x)
                )

                if len(annots_df[pd.isnull(annots_df.image_id)]) > 0:
                    warnings.warn(
                        "There are annotations in your dataset for which there is no matching images"
                        + f"(in split `{split}`). These annotations will be removed during any "
                        + "computation or conversion. It is recommended that you clean your dataset."
                    )

                master_df = pd.concat([master_df, annots_df], ignore_index=True)

        # Clean and finalize master DataFrame
        master_df = master_df[pd.notnull(master_df.image_id)]

        for col in ["image_width", "image_height", "class_id"]:
            master_df[col] = master_df[col].astype(np.int32)

        self.master_df = master_df

    def compute_bbox_and_area(self, seg):
        """Computes bounding box and area from RLE segmentation.

        Args:
            seg: RLE segmentation data

        Returns:
            Tuple containing:
                - List of bbox coordinates [x_min, y_min, width, height]
                - Integer area value
        """
        if isinstance(seg, dict) and "size" in seg and "counts" in seg:
            mask = maskUtils.decode(seg)  # Convert RLE to binary mask
            bbox = maskUtils.toBbox(
                seg
            ).tolist()  # Get bounding box [x_min, y_min, width, height]
            area = mask.sum()  # Count nonzero pixels
            return bbox, int(area)
        return [0, 0, 0, 0], 0  # Default values if segmentation is missing
