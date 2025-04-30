"""
This module provides utilities for converting between different dataset formats.

It includes functions and classes for:
- Converting between YOLO, COCO, Base, Mask and Detectron2 formats
- Handling both bounding box and segmentation annotations
- Preserving dataset splits and image organization
- Supporting parallel processing for performance

The main components are:

Classes:
    LabelEncoder: Maps category labels to numeric indices

Functions:
    convert_yolo: Converts to YOLO format
    convert_coco: Converts to COCO format
    convert_base: Converts to Base format
    seg_convert_mask: Converts to Mask format
    seg_convert_yolo: Converts to YOLO segmentation format
    seg_convert_coco: Converts to COCO segmentation format
    seg_convert_detectron: Converts to Detectron2 format

Helper Functions:
    _fastcopy: Parallel file copying
    _makedirs: Creates output directory structure
    write_yolo_txt: Writes YOLO format annotations
    write_base_json: Writes Base format annotations
    _make_coco_images/annotations/categories: Creates COCO format components
"""

import copy
import json
import os
import warnings
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from core_assist.dataset.utils import copyfile, ifnone, write_json
from joblib import Parallel, delayed
from tqdm.auto import tqdm


class LabelEncoder:
    """Maps category labels to numeric indices.

    Used to convert string category labels to integer indices required by some formats.
    Maintains a consistent mapping across multiple calls.
    """

    def __init__(self):
        self._map = dict()

    def fit(self, series):
        """Learns mapping from a series of labels.

        Args:
            series: Series of category labels
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        categories = series.unique().tolist()
        label_map = dict(zip(categories, np.arange(len(categories))))
        for k, _ in label_map.items():
            if k not in self._map:
                self._map[k] = label_map[k]

    def transform(self, series):
        """Converts labels to indices using learned mapping.

        Args:
            series: Series of category labels

        Returns:
            Series of numeric indices
        """
        series = series.map(self._map)
        return series

    def fit_transform(self, series):
        """Learns mapping and converts labels in one step.

        Args:
            series: Series of category labels

        Returns:
            Series of numeric indices
        """
        self.fit(series)
        return self.transform(series)


def _fastcopy(src_files: Union[str, os.PathLike], dest_dir: Union[str, os.PathLike]):
    """Copies files in parallel for better performance.

    Args:
        src_files: List of source file paths
        dest_dir: Destination directory
    """
    _ = Parallel(n_jobs=-1, backend="threading")(
        delayed(copyfile)(f, dest_dir) for f in src_files
    )


def _makedirs(
    src: Union[str, os.PathLike],
    ext: str,
    dest: Optional[Union[str, os.PathLike]] = None,
):
    """Creates output directory structure.

    Args:
        src: Source directory path
        ext: Extension/format name for output
        dest: Optional custom destination path

    Returns:
        Tuple of (image_dir, label_dir) paths
    """
    output_dir = ifnone(dest, src, Path)
    output_dir = output_dir / ext
    output_imagedir = output_dir / "images"
    output_labeldir = output_dir / "annotations"
    output_imagedir.mkdir(parents=True, exist_ok=True)
    output_labeldir.mkdir(parents=True, exist_ok=True)
    return output_imagedir, output_labeldir


def write_yolo_txt(
    filename: str, output_dir: Union[str, os.PathLike, PosixPath], yolo_string: str
):
    """Writes YOLO format annotations to a text file.

    Args:
        filename: Name of the image file
        output_dir: Output directory path
        yolo_string: YOLO format annotation string
    """
    filepath = Path(output_dir).joinpath(Path(filename).stem + ".txt")
    with open(filepath, "a") as f:
        f.write(yolo_string)
        f.write("\n")


def convert_yolo(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
):
    """Converts dataset to YOLO format.

    Converts bounding box annotations to YOLO format:
    <class_id> <x_center> <y_center> <width> <height>
    All values are normalized to [0,1]

    Args:
        df: Master DataFrame with annotations
        root: Root directory path
        copy_images: Whether to copy images
        save_under: Output subdirectory name
        output_dir: Custom output directory path
    """

    save_under = ifnone(save_under, "yolo")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    splits = df.split.unique().tolist()
    lbl = LabelEncoder()

    dataset = dict()

    for split in splits:
        output_subdir = output_labeldir / split if len(splits) > 0 else output_labeldir
        output_subdir.mkdir(parents=True, exist_ok=True)

        split_df = df.query("split == @split").copy()

        # drop images missing width or height information
        hw_missing = split_df[
            pd.isnull(split_df["image_width"]) | pd.isnull(split_df["image_height"])
        ]
        if len(hw_missing) > 0:
            warnings.warn(
                f"{hw_missing['image_id'].nunique()} has height/width information missing in split `{split}`. "
                + f"{len(hw_missing)} annotations will be removed."
            )

        split_df = split_df[
            pd.notnull(split_df["image_width"]) & pd.notnull(split_df["image_height"])
        ]

        split_df["x_center"] = split_df["x_min"] + split_df["width"] / 2
        split_df["y_center"] = split_df["y_min"] + split_df["height"] / 2

        # normalize
        split_df["x_center"] = split_df["x_center"] / split_df["image_width"]
        split_df["y_center"] = split_df["y_center"] / split_df["image_height"]
        split_df["width"] = split_df["width"] / split_df["image_width"]
        split_df["height"] = split_df["height"] / split_df["image_height"]

        split_df["class_index"] = lbl.fit_transform(split_df["category"])

        split_df["yolo_string"] = (
            split_df["class_index"].astype(str)
            + " "
            + split_df["x_center"].astype(str)
            + " "
            + split_df["y_center"].astype(str)
            + " "
            + split_df["width"].astype(str)
            + " "
            + split_df["height"].astype(str)
        )

        ds = (
            split_df.groupby("image_id")["yolo_string"]
            .agg(lambda x: "\n".join(x))
            .reset_index()
        )

        image_ids = ds["image_id"].tolist()
        yolo_strings = ds["yolo_string"].tolist()

        dataset[split] = str(Path(root) / "images" / split)

        for image_id, ystr in tqdm(
            zip(image_ids, yolo_strings), total=len(image_ids), desc=f"split: {split}"
        ):
            write_yolo_txt(image_id, output_subdir, ystr)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)

            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)

    dataset["nc"] = len(lbl._map)
    dataset["names"] = list(lbl._map.keys())

    with open(Path(output_labeldir).joinpath("dataset.yaml"), "w") as f:
        yaml.dump(dataset, f, default_flow_style=None, allow_unicode=True)


def _make_coco_images(df: pd.DataFrame, image_map: Dict) -> List:
    """Creates image list in COCO format.

    Args:
        df: DataFrame with image info
        image_map: Mapping of image IDs to indices

    Returns:
        List of image dictionaries in COCO format
    """
    df = copy.deepcopy(df)
    df.drop_duplicates(subset=["image_id"], keep="first", inplace=True)
    df = (
        df[["image_id", "image_height", "image_width"]]
        .copy()
        .rename(
            columns={
                "image_id": "file_name",
                "image_height": "height",
                "image_width": "width",
            }
        )
    )
    df["id"] = df["file_name"].map(image_map)
    df = df[["id", "file_name", "height", "width"]]
    image_list = list(df.to_dict(orient="index").values())
    return image_list


def _make_coco_annotations(df: pd.DataFrame, image_map: Dict) -> List:
    """Creates annotation list in COCO format.

    Args:
        df: DataFrame with annotations
        image_map: Mapping of image IDs to indices

    Returns:
        List of annotation dictionaries in COCO format
    """
    df = copy.deepcopy(df)
    df["bbox"] = df[["x_min", "y_min", "width", "height"]].apply(list, axis=1)
    df["area"] = df["height"] * df["width"]
    df.drop(
        ["x_min", "y_min", "width", "height", "image_width", "image_height"],
        axis=1,
        inplace=True,
    )
    df["id"] = range(len(df))
    df["image_id"] = df["image_id"].map(image_map)
    df.rename(columns={"class_id": "category_id"}, inplace=True)
    df["category_id"] = df["category_id"].astype(int)
    df["segmentation"] = [[]] * len(df)
    df["iscrowd"] = 0
    df = df[
        ["id", "image_id", "category_id", "bbox", "area", "segmentation", "iscrowd"]
    ].copy()
    annotation_list = list(df.to_dict(orient="index").values())
    return annotation_list


def _make_coco_categories(df: pd.DataFrame) -> List:
    """Creates category list in COCO format.

    Args:
        df: DataFrame with category info

    Returns:
        List of category dictionaries in COCO format
    """
    df = copy.deepcopy(df)
    df = (
        df.drop_duplicates(subset=["category"], keep="first")
        .sort_values("class_id")[["class_id", "category"]]
        .rename(columns={"category": "name", "class_id": "id"})
    )
    df["id"] = df["id"].astype(int)
    df["supercategory"] = "none"
    category_list = list(df.to_dict(orient="index").values())
    return category_list


def convert_coco(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
) -> None:
    """Converts dataset to COCO format.

    Creates COCO format JSON files with:
    - images: List of image info
    - annotations: List of bounding box annotations
    - categories: List of category definitions

    Args:
        df: Master DataFrame with annotations
        root: Root directory path
        copy_images: Whether to copy images
        save_under: Output subdirectory name
        output_dir: Custom output directory path
    """

    save_under = ifnone(save_under, "coco")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    splits = df.split.unique().tolist()

    for split in splits:
        split_df = df.query("split == @split").copy()

        split_df["image_path"] = split_df["image_path"].apply(lambda x: str(x))
        images = df["image_id"].unique().tolist()

        image_map = dict(zip(images, range(len(images))))
        image_list = _make_coco_images(split_df, image_map)
        annotation_list = _make_coco_annotations(split_df, image_map)
        category_list = _make_coco_categories(split_df)

        coco_dict = dict()
        coco_dict["images"] = image_list
        coco_dict["annotations"] = annotation_list
        coco_dict["categories"] = category_list
        output_file = (
            output_labeldir / f"{split}.json"
            if split != ""
            else output_labeldir / "annotations.json"
        )
        write_json(coco_dict, output_file)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)

            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)


def write_base_json(image_id, output_subdir, ystr, height, width):
    """Writes annotations in Base format JSON.

    Args:
        image_id: ID of the image
        output_subdir: Output directory path
        ystr: Annotation string
        height: Image height
        width: Image width
    """
    base_json = {
        "image_path": image_id,
        "image_name": Path(image_id).stem,
        "height": height,
        "width": width,
        "annotations": [],
    }

    for line in ystr.split("\n"):
        parts = line.split()
        if len(parts) == 5:
            label, x_min, y_min, w, h = parts
            base_json["annotations"].append(
                {
                    "label": label,
                    "bbox": [
                        int(float(x_min)),
                        int(float(y_min)),
                        int(float(w) + float(x_min)),
                        int(float(h) + float(y_min)),
                    ],
                }
            )

    json_path = output_subdir / f"{Path(image_id).stem}.json"
    with open(json_path, "w") as f:
        json.dump(base_json, f, indent=4)


def convert_base(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
):
    """Converts dataset to Base format.

    Creates JSON files with:
    - Image metadata
    - Bounding box annotations
    One JSON file per image.

    Args:
        df: Master DataFrame with annotations
        root: Root directory path
        copy_images: Whether to copy images
        save_under: Output subdirectory name
        output_dir: Custom output directory path
    """

    save_under = ifnone(save_under, "base")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)
    splits = df.split.unique().tolist()

    for split in splits:
        output_subdir = output_labeldir / split if len(splits) > 0 else output_labeldir
        output_subdir.mkdir(parents=True, exist_ok=True)
        split_df = df.query("split == @split").copy()

        # Drop images missing width or height information
        hw_missing = split_df[
            pd.isnull(split_df["image_width"]) | pd.isnull(split_df["image_height"])
        ]
        if len(hw_missing) > 0:
            warnings.warn(
                f"{hw_missing['image_path'].nunique()} images have missing height/width info in split `{split}`."
            )
        split_df = split_df[
            pd.notnull(split_df["image_width"]) & pd.notnull(split_df["image_height"])
        ]

        split_df["base_json"] = (
            split_df["category"].astype(str)
            + " "
            + split_df["x_min"].astype(str)
            + " "
            + split_df["y_min"].astype(str)
            + " "
            + split_df["width"].astype(str)
            + " "
            + split_df["height"].astype(str)
        )

        ds = (
            split_df.groupby("image_path")
            .agg(
                {
                    "base_json": lambda x: "\n".join(x),
                    "image_width": "first",
                    "image_height": "first",
                }
            )
            .reset_index()
        )

        image_ids = ds["image_path"].tolist()
        image_ids = [str(image_id) for image_id in image_ids]
        base_strings = ds["base_json"].tolist()
        widths = ds["image_width"].tolist()
        heights = ds["image_height"].tolist()

        for image_id, ystr, w, h in tqdm(
            zip(image_ids, base_strings, widths, heights),
            total=len(image_ids),
            desc=f"Processing split: {split}",
        ):
            write_base_json(image_id, output_subdir, ystr, h, w)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)
            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)


def seg_write_base_json(image_id, output_subdir, ystr, height, width):
    """Writes segmentation annotations in Base format JSON.

    Args:
        image_id: ID of the image
        output_subdir: Output directory path
        ystr: Annotation string
        height: Image height
        width: Image width
    """
    base_json = {
        "image_path": image_id,
        "image_name": Path(image_id).stem,
        "height": height,
        "width": width,
        "annotations": [],
    }

    for line in ystr.split("\n"):
        parts = line.split(maxsplit=1)  # Split into label and segmentation
        if len(parts) < 2:
            continue  # Skip invalid lines

        label = parts[0]
        segmentation_str = parts[1]

        try:
            # Check if segmentation is in JSON format (RLE)
            segmentation = json.loads(segmentation_str)
        except json.JSONDecodeError:
            # Otherwise, treat it as polygon data
            segmentation = list(map(int, map(float, segmentation_str.split())))

        base_json["annotations"].append({"label": label, "segmentation": segmentation})

    json_path = output_subdir / f"{Path(image_id).stem}.json"
    with open(json_path, "w") as f:
        json.dump(base_json, f, indent=4)


def seg_convert_base(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
):
    """Converts dataset to Base format with segmentation.

    Creates JSON files with:
    - Image metadata
    - Segmentation annotations (RLE or polygon format)
    One JSON file per image.

    Args:
        df: Master DataFrame with annotations
        root: Root directory path
        copy_images: Whether to copy images
        save_under: Output subdirectory name
        output_dir: Custom output directory path
    """

    save_under = ifnone(save_under, "base")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)
    splits = df.split.unique().tolist()

    for split in splits:
        output_subdir = output_labeldir / split if len(splits) > 0 else output_labeldir
        output_subdir.mkdir(parents=True, exist_ok=True)
        split_df = df.query("split == @split").copy()

        # Drop images missing width or height information
        hw_missing = split_df[
            pd.isnull(split_df["image_width"]) | pd.isnull(split_df["image_height"])
        ]
        if len(hw_missing) > 0:
            warnings.warn(
                f"{hw_missing['image_path'].nunique()} images have missing height/width info in split `{split}`."
            )
        split_df = split_df[
            pd.notnull(split_df["image_width"]) & pd.notnull(split_df["image_height"])
        ]

        # Convert segmentation into a JSON string if it's an RLE dictionary
        def format_segmentation(seg):
            if isinstance(seg, dict) and "size" in seg and "counts" in seg:
                return json.dumps(seg)  # Store as JSON string
            return " ".join(map(str, seg))  # Handle polygon format

        split_df["base_json"] = split_df.apply(
            lambda row: f"{row['category']} {format_segmentation(row['segmentation'])}",
            axis=1,
        )

        ds = (
            split_df.groupby("image_path")
            .agg(
                {
                    "base_json": lambda x: "\n".join(x),
                    "image_width": "first",
                    "image_height": "first",
                }
            )
            .reset_index()
        )

        image_ids = ds["image_path"].tolist()
        image_ids = [str(image_id) for image_id in image_ids]
        base_strings = ds["base_json"].tolist()
        widths = ds["image_width"].tolist()
        heights = ds["image_height"].tolist()

        for image_id, ystr, w, h in tqdm(
            zip(image_ids, base_strings, widths, heights),
            total=len(image_ids),
            desc=f"Processing split: {split}",
        ):
            seg_write_base_json(image_id, output_subdir, ystr, h, w)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)
            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)


def seg_write_yolo_txt(
    filename: str, output_dir: Union[str, os.PathLike, PosixPath], yolo_string: str
):
    """Writes YOLO segmentation annotations to a text file.

    Args:
        filename: Name of the image file
        output_dir: Output directory path
        yolo_string: YOLO format segmentation string
    """
    filepath = Path(output_dir).joinpath(Path(filename).stem + ".txt")
    with open(filepath, "a") as f:
        f.write(yolo_string)
        f.write("\n")


from pycocotools import mask as maskUtils


def seg_convert_mask(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
):
    """Converts dataset to binary mask format.

    Creates binary mask images where pixel values represent class IDs.
    One mask image per input image.

    Args:
        df: Master DataFrame with annotations
        root: Root directory path
        copy_images: Whether to copy images
        save_under: Output subdirectory name
        output_dir: Custom output directory path
    """

    save_under = ifnone(save_under, "mask")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    splits = df.split.unique().tolist()
    for split in splits:
        output_subdir = output_labeldir / split if len(splits) > 0 else output_labeldir
        output_subdir.mkdir(parents=True, exist_ok=True)

        split_df = df.query("split == @split").copy()

        for image_id, group in tqdm(
            split_df.groupby("image_id"), desc=f"Processing {split}"
        ):
            image_width = group.iloc[0]["image_width"]
            image_height = group.iloc[0]["image_height"]
            mask = np.zeros((image_height, image_width), dtype=np.uint8)

            for _, row in group.iterrows():
                if isinstance(row["segmentation"], dict):  # Ensure valid RLE format
                    rle = row["segmentation"]
                    binary_mask = maskUtils.decode(rle)
                    class_id = row["class_id"]
                    mask[binary_mask == 1] = class_id

            mask_path = output_subdir / f"{image_id}.png"
            cv2.imwrite(str(mask_path), mask)
            # print(f"Saved mask: {mask_path}")

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)

            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)


import cv2


def seg_convert_yolo(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
):
    """Converts dataset to YOLO segmentation format.

    Creates text files with:
    <class_id> <x1> <y1> <x2> <y2> ...
    Where x,y are normalized polygon coordinates.

    Args:
        df: Master DataFrame with annotations
        root: Root directory path
        copy_images: Whether to copy images
        save_under: Output subdirectory name
        output_dir: Custom output directory path
    """

    save_under = ifnone(save_under, "yolo")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    splits = df.split.unique().tolist()
    lbl = LabelEncoder()

    dataset = dict()

    for split in splits:
        output_subdir = output_labeldir / split if len(splits) > 0 else output_labeldir
        output_subdir.mkdir(parents=True, exist_ok=True)

        split_df = df.query("split == @split").copy()

        # Drop images missing width or height information
        hw_missing = split_df[
            pd.isnull(split_df["image_width"]) | pd.isnull(split_df["image_height"])
        ]
        if len(hw_missing) > 0:
            warnings.warn(
                f"{hw_missing['image_id'].nunique()} has height/width information missing in split `{split}`. "
                + f"{len(hw_missing)} annotations will be removed."
            )

        split_df = split_df[
            pd.notnull(split_df["image_width"]) & pd.notnull(split_df["image_height"])
        ]

        def convert_rle_to_polygon(row):
            """Converts RLE to polygon coordinates using the correct method."""
            if (
                isinstance(row["segmentation"], dict)
                and "size" in row["segmentation"]
                and "counts" in row["segmentation"]
            ):
                width = row["image_width"]
                height = row["image_height"]
                polygons = rle_to_polygon(row["segmentation"], width, height)

                # Flatten the nested list of polygons for YOLO format
                if polygons:
                    # Take the first polygon if multiple exist (YOLO format limitation)
                    # or concat multiple polygons with separation
                    flat_poly = []
                    for poly in polygons:
                        flat_poly.extend(poly)
                    return flat_poly
            elif isinstance(row["segmentation"], list):
                # If segmentation is already in polygon format, normalize it
                poly = [coord for segment in row["segmentation"] for coord in segment]
                norm_poly = [
                    (
                        poly[i] / row["image_width"]
                        if i % 2 == 0
                        else poly[i] / row["image_height"]
                    )
                    for i in range(len(poly))
                ]
                return norm_poly
            return []

        # Apply the conversion and create the normalized polygon strings
        split_df["polygon"] = split_df.apply(convert_rle_to_polygon, axis=1)

        # Convert polygon coordinates to YOLO format string
        def format_polygon_for_yolo(polygon):
            if polygon and len(polygon) > 0:
                return " ".join(map(str, polygon))
            return ""

        split_df["normalized_polygon"] = split_df["polygon"].apply(
            format_polygon_for_yolo
        )

        split_df["class_index"] = lbl.fit_transform(split_df["category"])

        # Filter out rows with empty polygons
        split_df = split_df[split_df["normalized_polygon"] != ""]

        split_df["yolo_string"] = (
            split_df["class_index"].astype(str) + " " + split_df["normalized_polygon"]
        )

        ds = (
            split_df.groupby("image_id")["yolo_string"]
            .agg(lambda x: "\n".join(x))
            .reset_index()
        )

        image_ids = ds["image_id"].tolist()
        yolo_strings = ds["yolo_string"].tolist()

        dataset[split] = str(Path(root) / "images" / split)

        for image_id, ystr in tqdm(
            zip(image_ids, yolo_strings), total=len(image_ids), desc=f"split: {split}"
        ):
            seg_write_yolo_txt(image_id, output_subdir, ystr)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)

            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)

    dataset["nc"] = len(lbl._map)
    dataset["names"] = list(lbl._map.keys())

    with open(Path(output_labeldir).joinpath("dataset.yaml"), "w") as f:
        yaml.dump(dataset, f, default_flow_style=None, allow_unicode=True)


def rle_to_polygon(rle, width, height):
    """
    Convert RLE mask to polygon coordinates.

    Args:
        rle (dict): RLE mask with 'size' and 'counts' keys.
        width (int): Image width.
        height (int): Image height.

    Returns:
        list: List of normalized polygon coordinates.
    """
    # Decode RLE to binary mask
    binary_mask = maskUtils.decode(rle)

    # Find contours from the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Simplify contours to polygons
    polygons = []
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Normalize polygon coordinates to [0, 1]
        polygon = approx.flatten().astype(float)
        polygon[::2] /= width  # Normalize x coordinates
        polygon[1::2] /= height  # Normalize y coordinates
        polygons.append(polygon.tolist())

    return polygons


from pycocotools import mask as maskUtils


def _make_seg_coco_annotations(df: pd.DataFrame, image_map: Dict) -> List:
    """Creates annotation list for COCO with segmentation in RLE format."""
    df = copy.deepcopy(df)
    df["image_id"] = df["image_id"].map(image_map)
    df["id"] = range(len(df))
    df.rename(columns={"class_id": "category_id"}, inplace=True)
    df["category_id"] = df["category_id"].astype(int)

    def compute_bbox_and_area(seg):
        """Computes bounding box and area from RLE segmentation."""
        if isinstance(seg, dict) and "size" in seg and "counts" in seg:
            mask = maskUtils.decode(seg)  # Convert RLE to binary mask
            bbox = maskUtils.toBbox(
                seg
            ).tolist()  # Get bounding box [x_min, y_min, width, height]
            area = mask.sum()  # Count nonzero pixels
            return bbox, int(area)
        return [0, 0, 0, 0], 0  # Default values if segmentation is missing

    tqdm.pandas(desc="Processing RLE masks")
    df["bbox"], df["area"] = zip(*df["segmentation"].map(compute_bbox_and_area))

    df["iscrowd"] = 1  # RLE segmentation should always have iscrowd = 1

    df = df[
        ["id", "image_id", "category_id", "bbox", "area", "segmentation", "iscrowd"]
    ].copy()
    annotation_list = list(df.to_dict(orient="index").values())

    return annotation_list


def seg_convert_coco(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
) -> None:
    """Converts to COCO format from the master dataframe, keeping RLE segmentation."""

    save_under = ifnone(save_under, "coco")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    splits = df.split.unique().tolist()

    for split in splits:
        split_df = df.query("split == @split").copy()

        # split_df["file_name"] = split_df["file_name"]
        # .apply(lambda x: str(x))
        images = df["image_id"].unique().tolist()

        image_map = dict(zip(images, range(len(images))))
        image_list = _make_coco_images(split_df, image_map)
        annotation_list = _make_seg_coco_annotations(split_df, image_map)
        category_list = _make_coco_categories(split_df)

        coco_dict = {
            "images": image_list,
            "annotations": annotation_list,
            "categories": category_list,
        }

        output_file = (
            output_labeldir / f"{split}.json"
            if split != ""
            else output_labeldir / "annotations.json"
        )
        tqdm.write(f"Saving JSON: {output_file}")
        write_json(coco_dict, output_file)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)
            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)


def _rle_to_polygons(rle: Dict) -> List[List[float]]:
    """
    Converts RLE segmentation to polygon format.

    Args:
        rle (Dict): RLE segmentation dictionary with keys "size" and "counts".

    Returns:
        List[List[float]]: List of polygons, where each polygon is a list of coordinates [x1, y1, x2, y2, ...].
    """
    if not isinstance(rle, dict) or "size" not in rle or "counts" not in rle:
        return []

    # Decode RLE to binary mask
    mask = maskUtils.decode(rle)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to polygons
    polygons = []
    for contour in contours:
        if contour.size >= 6:  # At least 3 points (x, y) to form a polygon
            contour = contour.astype(np.float32).reshape(-1, 2).tolist()
            polygons.append(contour)

    return polygons


def _make_detectron_images(df: pd.DataFrame, image_map: Dict) -> List[Dict]:
    """Creates image list for Detectron2 format."""
    image_list = []
    for _, row in df.drop_duplicates(subset=["image_id"]).iterrows():
        image_dict = {
            "file_name": str(row["image_path"]),
            "height": int(row["image_height"]),
            "width": int(row["image_width"]),
            "image_id": image_map[row["image_id"]],  # Map image_id to a unique integer
        }
        image_list.append(image_dict)
    return image_list


def _make_detectron_annotations(df: pd.DataFrame, image_map: Dict) -> List[Dict]:
    """Creates annotation list for Detectron2 format."""
    annotation_list = []
    for idx, row in df.iterrows():
        # Convert RLE segmentation to polygons if necessary
        segmentation = row["segmentation"]
        if isinstance(segmentation, dict):  # RLE format
            segmentation = _rle_to_polygons(segmentation)

        annotation_dict = {
            "bbox": [
                float(row["x_min"]),
                float(row["y_min"]),
                float(row["width"]),
                float(row["height"]),
            ],
            "bbox_mode": 1,  # COCO format
            "category_id": int(row["class_id"]),
            "segmentation": segmentation,  # Polygon format
            "keypoints": [],  # Not used
            "iscrowd": 0,  # Assuming no crowd annotations
            "image_id": image_map[row["image_id"]],  # Map image_id to a unique integer
        }
        annotation_list.append(annotation_dict)
    return annotation_list


def seg_convert_detectron(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, Path],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, Path]] = None,
) -> None:
    """
    Converts the master DataFrame to Detectron2 format, converting RLE segmentation to polygons.

    Args:
        df (pd.DataFrame): Master DataFrame with columns:
            - image_id, image_width, image_height, segmentation, class_id, image_path, split, x_min, y_min, width, height, area.
        root (Union[str, os.PathLike, Path]): Root directory of the dataset.
        copy_images (bool): Whether to copy images to the output directory.
        save_under (Optional[str]): Subdirectory to save the output files.
        output_dir (Optional[Union[str, os.PathLike, Path]]): Custom output directory.
    """
    save_under = ifnone(save_under, "detectron2")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    splits = df["split"].unique().tolist()

    for split in splits:
        split_df = df.query("split == @split").copy()

        # Create a mapping from original image_id to a unique integer
        images = split_df["image_id"].unique().tolist()
        image_map = dict(zip(images, range(len(images))))

        # Create Detectron2-compatible image and annotation lists
        image_list = _make_detectron_images(split_df, image_map)
        annotation_list = _make_detectron_annotations(split_df, image_map)

        # Combine images and annotations into Detectron2 format
        detectron_data = []
        for image_dict in image_list:
            image_id = image_dict["image_id"]
            annotations = [
                ann for ann in annotation_list if ann["image_id"] == image_id
            ]
            image_dict["annotations"] = annotations
            detectron_data.append(image_dict)

        # Save the Detectron2 JSON file
        output_file = (
            output_labeldir / f"{split}.json"
            if split != ""
            else output_labeldir / "annotations.json"
        )
        tqdm.write(f"Saving JSON: {output_file}")
        with open(output_file, "w") as f:
            json.dump(detectron_data, f, indent=4)

        # Copy images if requested
        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)
            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)
