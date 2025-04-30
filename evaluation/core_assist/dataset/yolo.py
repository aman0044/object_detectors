import os
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union

import imagesize
import numpy as np
import pandas as pd
import yaml
from core_assist.dataset.format import FormatSpec, SegFormatSpec
from core_assist.dataset.utils import exists, get_annotation_dir, get_image_dir
from joblib import Parallel, delayed
from tqdm.auto import tqdm

NUM_THREADS = os.cpu_count() // 2
import cv2
from pycocotools import mask as maskUtils
from pycocotools import mask as mask_util

"""
This module provides classes for working with YOLO format annotations.

The YOLO format is a common annotation format used in object detection and segmentation tasks.
It includes two main classes:
- Yolo: For handling bounding box annotations in YOLO format
- SegmentationYolo: For handling segmentation annotations in YOLO format

The module supports loading annotations from either:
1. A directory structure with images and annotation files
2. A CSV file containing paths to images and annotations

Key features:
- Parallel processing for faster annotation loading
- Support for dataset splits (train/val/test)
- Automatic conversion between normalized YOLO coordinates and absolute pixel coordinates
- Integration with category mapping from YAML files
"""


class Yolo(FormatSpec):
    """Represents a YOLO annotation object.

    Args:
        root (Union[str, os.PathLike]): path to root directory or CSV file.
            - If root is a directory, expects either of the following layouts:
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
                        │   ├── 1.txt
                        │   ├── 2.txt
                        │   │   ...
                        │   └── n.txt
                        ├── valid (...)
                        ├── test (...)
                        └── dataset.yaml [Optional]

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
                        ├── 1.txt
                        ├── 2.txt
                        │ ...
                        ├── n.txt
                        └── dataset.yaml [Optional]

            - If root is a CSV file, expects columns:
              - img_path: path to image file
              - anno_path: path to annotation file
              - split (optional): dataset split (train, valid, test, etc.)

        format (Optional[str]): Format specification for the annotations. Defaults to None.
        mapping (Optional[Dict]): Dictionary mapping class IDs to category names. If not provided,
            will attempt to load from dataset.yaml file.

    Attributes:
        root (Union[str, os.PathLike]): Path to root directory or CSV file
        mapping (Optional[Dict]): Dictionary mapping class IDs to category names
        master_df (pd.DataFrame): DataFrame containing all annotations with columns:
            - split: Dataset split (train/valid/test)
            - image_id: Image filename
            - image_width: Width of image in pixels
            - image_height: Height of image in pixels
            - x_min: Left coordinate of bounding box
            - y_min: Top coordinate of bounding box
            - width: Width of bounding box
            - height: Height of bounding box
            - category: Category name or ID
            - image_path: Full path to image file
    """

    def __init__(
        self, root: Union[str, os.PathLike], format: Optional[str] = None, mapping=None
    ):
        self.root = root
        super().__init__(root, format=format)
        # self.is_csv = is_csv
        self.mapping = mapping
        # print(f"Mapping: {self.mapping}")
        # Check if root is a CSV file
        try:
            if str(root).endswith(".csv"):
                self.csv_df = pd.read_csv(root)
                if all(col in self.csv_df.columns for col in ["img_path", "anno_path"]):
                    self.is_csv = True
            else:
                self.is_csv = False
        except Exception:
            pass

        if not self.is_csv:
            # Root is a directory
            self.class_file = [y for y in Path(self.root).glob("*.yaml")]
            self._image_dir = get_image_dir(root)
            self._annotation_dir = get_annotation_dir(root)
            self._has_image_split = False
            assert exists(self._image_dir), "root is missing 'images' directory."
            assert exists(
                self._annotation_dir
            ), "root is missing 'annotations' directory."
            self._find_splits()
        else:
            # Root is a CSV
            if "split" in self.csv_df.columns:
                self._splits = self.csv_df["split"].unique().tolist()
            else:
                self._splits = ["main"]

        self._resolve_dataframe()

    def _resolve_dataframe(self):
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

        if self.is_csv:
            print("Loading annotations from CSV:")
            image_ids = []
            image_paths = []
            class_ids = []
            x_mins = []
            y_mins = []
            bbox_widths = []
            bbox_heights = []
            image_heights = []
            image_widths = []
            splits = []

            def process_row(row):
                img_path = row["img_path"]
                anno_path = row["anno_path"]
                split = row.get(
                    "split", "main"
                )  # Default to 'main' if split not provided

                # Verify files exist
                if not exists(img_path) or not exists(anno_path):
                    return None

                # Get image dimensions
                try:
                    im_width, im_height = imagesize.get(img_path)
                except Exception:
                    return None

                # Parse annotation file
                try:
                    with open(anno_path, "r") as f:
                        instances = f.read().strip().split("\n")
                        results = []
                        for ins in instances:
                            parts = ins.split()
                            if len(parts) >= 5:  # Ensure valid format
                                class_id, x, y, w, h = list(map(float, parts))
                                results.append(
                                    (
                                        Path(img_path).name,  # image_id
                                        img_path,  # image_path
                                        im_width,  # image_width
                                        im_height,  # image_height
                                        int(class_id),  # class_id
                                        max(float((x - w / 2) * im_width), 0),  # x_min
                                        max(float((y - h / 2) * im_height), 0),  # y_min
                                        float(w * im_width),  # width
                                        float(h * im_height),  # height
                                        split,  # split
                                    )
                                )
                        return results
                except Exception as e:
                    print(f"Error parsing {anno_path}: {e}")
                    return None

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_row, row)
                    for _, row in self.csv_df.iterrows()
                ]
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing CSV"
                ):
                    result = future.result()
                    if result:
                        for res in result:
                            (
                                image_id,
                                image_path,
                                image_width,
                                image_height,
                                class_id,
                                x_min,
                                y_min,
                                width,
                                height,
                                split,
                            ) = res
                            image_ids.append(image_id)
                            image_paths.append(image_path)
                            image_widths.append(image_width)
                            image_heights.append(image_height)
                            class_ids.append(class_id)
                            x_mins.append(x_min)
                            y_mins.append(y_min)
                            bbox_widths.append(width)
                            bbox_heights.append(height)
                            splits.append(split)

            # Create DataFrame from parsed data
            if image_ids:
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
                            splits,
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
                        "split",
                    ],
                )
                master_df = pd.concat([master_df, annots_df], ignore_index=True)

                # Apply category mapping if available
                if hasattr(self, "mapping") and self.mapping:
                    master_df["category"] = master_df["class_id"].map(self.mapping)
                else:
                    master_df["category"] = master_df["class_id"].astype(str)

        else:
            # Original directory-based processing
            print("Loading yolo annotations from directory:")
            for split in self._splits:
                image_ids = []
                image_paths = []
                class_ids = []
                x_mins = []
                y_mins = []
                bbox_widths = []
                bbox_heights = []
                image_heights = []
                image_widths = []

                split_path = split if self._has_image_split else ""
                annotations = (
                    Path(self._annotation_dir).joinpath(split_path).glob("*.txt")
                )
                parse_partial = partial(self._parse_txt_file, split_path)
                all_instances = Parallel(n_jobs=NUM_THREADS, backend="multiprocessing")(
                    delayed(parse_partial)(txt) for txt in tqdm(annotations, desc=split)
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

            # get category names from `dataset.yaml`
            try:
                with open(Path(self._annotation_dir).joinpath("dataset.yaml")) as f:
                    label_desc = yaml.load(f, Loader=yaml.FullLoader)

                categories = label_desc["names"]
                label_map = dict(zip(range(len(categories)), categories))
            except FileNotFoundError:
                label_map = dict()
                warnings.warn(f"No `dataset.yaml` file found in {self._annotation_dir}")

            master_df["class_id"] = master_df["class_id"].astype(np.int32)

            if label_map:
                master_df["category"] = master_df["class_id"].map(label_map)
            else:
                master_df["category"] = master_df["class_id"].astype(str)

        self.master_df = master_df

    def _parse_txt_file(self, split: str, txt: Union[str, os.PathLike]) -> Dict:
        """Parse txt annotations in yolo format

        Args:
            split (str): dataset split
            txt (Union[str, os.PathLike]): annotations file path

        Returns:
            Dict: dict containing scaled annotation for each line in the text file.
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
        stem = txt.stem
        try:
            img_file = list(Path(self._image_dir).joinpath(split).glob(f"{stem}*"))[0]
            im_width, im_height = imagesize.get(img_file)
        except IndexError:  # if the image file does not exist
            return label_info

        with open(txt, "r") as f:
            instances = f.read().strip().split("\n")
            for ins in instances:
                class_id, x, y, w, h = list(map(float, ins.split()))
                label_info["image_ids"].append(img_file.name)
                label_info["image_paths"].append(img_file)
                label_info["class_ids"].append(int(class_id))
                label_info["x_mins"].append(
                    max(float((float(x) - w / 2) * im_width), 0)
                )
                label_info["y_mins"].append(max(float((y - h / 2) * im_height), 0))
                label_info["bbox_widths"].append(float(w * im_width))
                label_info["bbox_heights"].append(float(h * im_height))
                label_info["image_widths"].append(im_width)
                label_info["image_heights"].append(im_height)
        return label_info


import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import imagesize
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from pycocotools import mask as mask_util
from tqdm import tqdm


class SegmentationYolo(SegFormatSpec):
    """Represents a YOLO segmentation annotation object.

    This class handles YOLO format segmentation annotations, supporting both polygon and RLE formats.
    It can load annotations from either a directory structure or a CSV file.

    Args:
        root (Union[str, os.PathLike]): Path to root directory or CSV file containing annotations
        format (Optional[str]): Format specification for the annotations. Defaults to None.
        mapping (Optional[Dict]): Dictionary mapping class IDs to category names

    Attributes:
        root (Union[str, os.PathLike]): Path to root directory or CSV file
        mapping (Dict): Dictionary mapping class IDs to category names
        master_df (pd.DataFrame): DataFrame containing all annotations with columns:
            - split: Dataset split (train/valid/test)
            - image_id: Image filename
            - image_width: Width of image in pixels
            - image_height: Height of image in pixels
            - segmentation: RLE encoded segmentation mask
            - category: Category name or ID
            - image_path: Full path to image file
            - x_min: Left coordinate of bounding box
            - y_min: Top coordinate of bounding box
            - width: Width of bounding box
            - height: Height of bounding box
            - area: Area of segmentation mask in pixels
    """

    def __init__(
        self, root: Union[str, os.PathLike], format: Optional[str] = None, mapping=None
    ):
        self.root = root
        super().__init__(root, format=format)
        self.mapping = mapping
        self.class_file = (
            [y for y in Path(self.root).glob("*.yaml")]
            if os.path.isdir(self.root)
            else []
        )
        self._image_dir = get_image_dir(root) if os.path.isdir(self.root) else None
        self._annotation_dir = (
            get_annotation_dir(root) if os.path.isdir(self.root) else None
        )
        self._has_image_split = False
        self._splits = []
        self.master_df = pd.DataFrame()

        if os.path.isdir(self.root):
            assert exists(self._image_dir), "root is missing 'images' directory."
            assert exists(
                self._annotation_dir
            ), "root is missing 'annotations' directory."
            self._find_splits()
            self._resolve_dataframe()
        elif os.path.isfile(self.root) and str(self.root).endswith(".csv"):
            self._resolve_dataframe_from_csv()
        else:
            raise ValueError("root must be either a directory or a CSV file.")

    def _resolve_dataframe_from_csv(self):
        """Resolve the master dataframe from a CSV file."""
        df = pd.read_csv(self.root)
        assert (
            "img_path" in df.columns and "anno_path" in df.columns
        ), "CSV must contain 'img_path' and 'anno_path' columns."
        assert (
            self.mapping is not None
        ), "Mapping is required to map class IDs to category names. Please provide a valid mapping dictionary."

        master_df = pd.DataFrame(
            columns=[
                "split",
                "image_id",
                "image_width",
                "image_height",
                "segmentation",
                "category",
                "image_path",
            ],
        )

        print("Loading yolo annotations from CSV:")
        image_ids = []
        image_paths = []
        class_ids = []
        segmentations = []
        image_heights = []
        image_widths = []
        splits = []
        x_mins = []
        y_mins = []
        widths = []
        heights = []
        areas = []

        def process_row(row):
            img_path = row["img_path"]
            anno_path = row["anno_path"]
            split = row.get("split", "main")

            try:
                im_width, im_height = imagesize.get(img_path)
            except Exception as e:
                warnings.warn(f"Failed to get image size for {img_path}: {e}")
                return None

            with open(anno_path, "r") as f:
                instances = f.read().strip().split("\n")
                results = []
                for ins in instances:
                    values = list(map(float, ins.split()))
                    class_id = int(values[0])
                    polygon = np.array(values[1:]).reshape(-1, 2)
                    polygon[:, 0] *= im_width
                    polygon[:, 1] *= im_height

                    # Convert polygon to RLE
                    mask = np.zeros((im_height, im_width), dtype=np.uint8)
                    polygon_int = polygon.astype(np.int32)
                    cv2.fillPoly(mask, [polygon_int], 1)
                    rle = mask_util.encode(np.asfortranarray(mask))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    bbox, area = self.compute_bbox_and_area(rle)
                    x, y, w, h = bbox

                    results.append(
                        (
                            Path(img_path).name,  # image_id
                            img_path,  # image_path
                            im_width,  # image_width
                            im_height,  # image_height
                            int(class_id),  # class_id
                            rle,  # segmentation
                            split,  # split
                            x,  # x_min
                            y,  # y_min
                            w,  # width
                            h,  # height
                            area,  # area
                        )
                    )
                return results

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing CSV"
            ):
                result = future.result()
                if result:
                    for res in result:
                        (
                            image_id,
                            image_path,
                            image_width,
                            image_height,
                            class_id,
                            segmentation,
                            split,
                            x_min,
                            y_min,
                            width,
                            height,
                            area,
                        ) = res
                        image_ids.append(image_id)
                        image_paths.append(image_path)
                        image_widths.append(image_width)
                        image_heights.append(image_height)
                        class_ids.append(class_id)
                        segmentations.append(segmentation)
                        splits.append(split)
                        x_mins.append(x_min)
                        y_mins.append(y_min)
                        widths.append(width)
                        heights.append(height)
                        areas.append(area)

        annots_df = pd.DataFrame(
            list(
                zip(
                    image_ids,
                    image_paths,
                    image_widths,
                    image_heights,
                    class_ids,
                    segmentations,
                    splits,
                    x_mins,
                    y_mins,
                    widths,
                    heights,
                    areas,
                )
            ),
            columns=[
                "image_id",
                "image_path",
                "image_width",
                "image_height",
                "class_id",
                "segmentation",
                "split",
                "x_min",
                "y_min",
                "width",
                "height",
                "area",
            ],
        )
        label_map = self.mapping
        annots_df["class_id"] = annots_df["class_id"].astype(np.int32)

        if label_map:
            annots_df["category"] = annots_df["class_id"].map(label_map)
        else:
            annots_df["category"] = annots_df["class_id"].astype(str)

        self.master_df = pd.concat([master_df, annots_df], ignore_index=True)

    def _resolve_dataframe(self):
        """Resolve the master dataframe from directory structure."""
        master_df = pd.DataFrame(
            columns=[
                "split",
                "image_id",
                "image_width",
                "image_height",
                "segmentation",
                "category",
                "image_path",
                "x_min",
                "y_min",
                "width",
                "height",
            ],
        )
        print("Loading yolo annotations:")
        for split in self._splits:
            image_ids = []
            image_paths = []
            class_ids = []
            segmentations = []
            image_heights = []
            image_widths = []
            x_mins = []
            y_mins = []
            widths = []
            heights = []
            area = []

            split = split if self._has_image_split else ""
            annotations = Path(self._annotation_dir).joinpath(split).glob("*.txt")

            parse_partial = partial(self._parse_txt_file, split)
            all_instances = Parallel(n_jobs=NUM_THREADS, backend="multiprocessing")(
                delayed(parse_partial)(txt) for txt in tqdm(annotations, desc=split)
            )
            for instances in all_instances:
                image_ids.extend(instances["image_ids"])
                image_paths.extend(instances["image_paths"])
                class_ids.extend(instances["class_ids"])
                segmentations.extend(instances["segmentations"])
                image_widths.extend(instances["image_widths"])
                image_heights.extend(instances["image_heights"])
                x_mins.extend(instances["x_mins"])
                y_mins.extend(instances["y_mins"])
                widths.extend(instances["widths"])
                heights.extend(instances["heights"])
                area.extend(instances["area"])

            annots_df = pd.DataFrame(
                list(
                    zip(
                        image_ids,
                        image_paths,
                        image_widths,
                        image_heights,
                        class_ids,
                        segmentations,
                        x_mins,
                        y_mins,
                        widths,
                        heights,
                        area,
                    )
                ),
                columns=[
                    "image_id",
                    "image_path",
                    "image_width",
                    "image_height",
                    "class_id",
                    "segmentation",
                    "x_min",
                    "y_min",
                    "width",
                    "height",
                    "area",
                ],
            )
            annots_df["split"] = split if split else "main"
            master_df = pd.concat([master_df, annots_df], ignore_index=True)

        # get category names from `dataset.yaml`
        try:
            with open(Path(self._annotation_dir).joinpath("dataset.yaml")) as f:
                label_desc = yaml.load(f, Loader=yaml.FullLoader)

            categories = label_desc["names"]
            label_map = dict(zip(range(len(categories)), categories))
        except FileNotFoundError:
            label_map = dict()
            warnings.warn(f"No `dataset.yaml` file found in {self._annotation_dir}")

        master_df["class_id"] = master_df["class_id"].astype(np.int32)

        if label_map:
            master_df["category"] = master_df["class_id"].map(label_map)
        else:
            master_df["category"] = master_df["class_id"].astype(str)
        self.master_df = master_df

    def _parse_txt_file(self, split: str, txt: Union[str, os.PathLike]) -> Dict:
        """Parse txt annotations in YOLO segmentation format.

        Processes a YOLO format text file containing segmentation annotations.
        Each line in the file represents one instance with format:
        <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
        where coordinates are normalized to [0,1].

        Args:
            split (str): Dataset split name (e.g. 'train', 'valid', 'test')
            txt (Union[str, os.PathLike]): Path to annotation text file

        Returns:
            Dict: Dictionary containing parsed annotations with keys:
                - image_ids: List of image filenames
                - image_paths: List of full image paths
                - class_ids: List of class IDs
                - segmentations: List of RLE encoded segmentation masks
                - image_heights: List of image heights
                - image_widths: List of image widths
                - x_mins: List of bounding box left coordinates
                - y_mins: List of bounding box top coordinates
                - widths: List of bounding box widths
                - heights: List of bounding box heights
                - area: List of segmentation mask areas
        """
        label_info_keys = [
            "image_ids",
            "image_paths",
            "class_ids",
            "segmentations",
            "image_heights",
            "image_widths",
            "x_mins",
            "y_mins",
            "widths",
            "heights",
            "area",
        ]
        label_info = {key: [] for key in label_info_keys}
        stem = txt.stem
        try:
            img_file = list(Path(self._image_dir).joinpath(split).glob(f"{stem}*"))[0]
            im_width, im_height = imagesize.get(img_file)
        except IndexError:  # if the image file does not exist
            return label_info

        with open(txt, "r") as f:
            instances = f.read().strip().split("\n")
            for ins in instances:
                values = list(map(float, ins.split()))
                class_id = int(values[0])
                polygon = np.array(values[1:]).reshape(-1, 2)
                polygon[:, 0] *= im_width
                polygon[:, 1] *= im_height

                # Convert polygon to RLE
                mask = np.zeros((im_height, im_width), dtype=np.uint8)
                polygon_int = polygon.astype(np.int32)
                cv2.fillPoly(mask, [polygon_int], 1)
                rle = mask_util.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode(
                    "utf-8"
                )  # Ensure RLE counts are JSON serializable
                bbox, area = self.compute_bbox_and_area(rle)
                x, y, w, h = bbox

                label_info["image_ids"].append(img_file.name)
                label_info["image_paths"].append(img_file)
                label_info["segmentations"].append(rle)
                label_info["class_ids"].append(int(class_id))
                label_info["image_widths"].append(im_width)
                label_info["image_heights"].append(im_height)
                label_info["x_mins"].append(x)
                label_info["y_mins"].append(y)
                label_info["widths"].append(w)
                label_info["heights"].append(h)
                label_info["area"].append(area)
        return label_info

    def compute_bbox_and_area(self, seg):
        """Computes bounding box and area from RLE segmentation.

        Args:
            seg (Dict): RLE segmentation encoding with keys 'size' and 'counts'

        Returns:
        Computes bounding box and area from RLE segmentation."""
        if isinstance(seg, dict) and "size" in seg and "counts" in seg:
            mask = maskUtils.decode(seg)  # Convert RLE to binary mask
            bbox = maskUtils.toBbox(
                seg
            ).tolist()  # Get bounding box [x_min, y_min, width, height]
            area = mask.sum()  # Count nonzero pixels
            return bbox, int(area)
        return [0, 0, 0, 0], 0
