import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from core_assist.dataset.format import SegFormatSpec
from pycocotools import mask as maskUtils
from tqdm import tqdm


class SegmentationMask(SegFormatSpec):
    """Represents a Mask-based segmentation annotation object.

    This class processes segmentation data and stores it in a master dataframe,
    ensuring the segmentation is in RLE format. It supports both single-class and
    multi-class segmentation masks, handling background classes appropriately.

    Args:
        root (Union[str, os.PathLike]): Path to CSV file containing image and mask paths
        format (Optional[str]): Format specification for the annotations
        mapping (Optional[dict]): Dictionary mapping class IDs to class names

    The CSV file must contain:
        - img_path: Path to the image file
        - mask_path: Path to the corresponding segmentation mask
        - splits (optional): Dataset split (train/val/test)

    The segmentation masks should be grayscale images where:
        - 0 typically represents background
        - Other integer values represent different classes
    """

    def __init__(
        self, root: Union[str, os.PathLike], format: Optional[str] = None, mapping=None
    ):
        super().__init__(root, format=format)
        self._is_csv = False
        self.mapping = mapping or {}  # Default to empty dict if mapping is None

        if isinstance(root, (str, os.PathLike)) and str(root).endswith(".csv"):
            self._is_csv = True
            self._csv_data = pd.read_csv(root)
            assert (
                "img_path" in self._csv_data.columns
            ), "CSV must have 'img_path' column."
            assert (
                "mask_path" in self._csv_data.columns
            ), "CSV must have 'mask_path' column."
            self._has_splits = "splits" in self._csv_data.columns
        else:
            raise ValueError("Please provide a valid CSV path")

        self._resolve_dataframe()

    def _resolve_dataframe(self):
        """Processes the input CSV file to create a master dataframe with segmentation information.

        This method:
        1. Reads each mask file
        2. Identifies unique classes in each mask
        3. Converts segmentations to RLE format
        4. Computes bounding boxes and areas
        5. Processes images in parallel for efficiency

        The resulting dataframe contains:
            - image_id: Unique identifier for each image
            - image_width/height: Dimensions of the image
            - segmentation: RLE encoded segmentation mask
            - category: Class name from mapping or class ID
            - class_id: Integer class identifier
            - image_path: Path to the image file
            - split: Dataset split if provided
            - x_min, y_min, width, height: Bounding box coordinates
            - area: Area of the segmentation mask
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
            # Create a list to store all results
            all_results = []

            def process_row(row_idx, row):
                try:
                    img_path = Path(row["img_path"])
                    mask_path = Path(row["mask_path"])
                    split = row["splits"] if self._has_splits else ""

                    # Read mask
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        warnings.warn(
                            f"Could not read mask at {mask_path}. Skipping row {row_idx}."
                        )
                        return []

                    image_height, image_width = mask.shape[:2]
                    results = []
                    unique_classes = np.unique(mask)

                    # If the only value is 0 (background), we should still include the image with empty segmentation
                    if len(unique_classes) == 1 and unique_classes[0] == 0:
                        data = {
                            "image_id": img_path.stem,
                            "image_width": image_width,
                            "image_height": image_height,
                            "segmentation": None,
                            "category": "background",
                            "class_id": 0,
                            "image_path": str(img_path),
                            "split": split,
                            "x_min": 0,
                            "y_min": 0,
                            "width": 0,
                            "height": 0,
                            "area": 0,
                        }
                        results.append(data)
                    else:
                        for class_id in unique_classes:
                            # Skip background class (usually 0) if there are other classes
                            if class_id == 0 and len(unique_classes) > 1:
                                continue

                            # Create binary mask for this class
                            binary_mask = np.zeros_like(mask, dtype=np.uint8)
                            binary_mask[mask == class_id] = 1

                            # Check if mask is empty
                            if np.sum(binary_mask) == 0:
                                continue

                            # Convert to RLE format
                            rle_segmentation = maskUtils.encode(
                                np.asfortranarray(binary_mask)
                            )
                            rle_segmentation["counts"] = rle_segmentation[
                                "counts"
                            ].decode("utf-8")

                            bbox, area = self.compute_bbox_and_area(rle_segmentation)

                            # Get category name from mapping
                            category = self.mapping.get(int(class_id), str(class_id))

                            data = {
                                "image_id": img_path.stem,
                                "image_width": image_width,
                                "image_height": image_height,
                                "segmentation": rle_segmentation,
                                "category": category,
                                "class_id": int(class_id),
                                "image_path": str(img_path),
                                "split": split,
                                "x_min": bbox[0],
                                "y_min": bbox[1],
                                "width": bbox[2],
                                "height": bbox[3],
                                "area": area,
                            }
                            results.append(data)

                    return results
                except Exception as e:
                    warnings.warn(f"Error processing row {row_idx}: {e}")
                    return []

            # Process in parallel
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_row, idx, row)
                    for idx, (_, row) in enumerate(self._csv_data.iterrows())
                ]

                # Use tqdm for progress tracking
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing masks"
                ):
                    result = future.result()
                    all_results.extend(result)

            # Check if we have any results
            if not all_results:
                warnings.warn(
                    "No valid segmentation data was found in the provided CSV."
                )

            # Convert all results to DataFrame in one operation (more efficient)
            if all_results:
                master_df = pd.concat(
                    [master_df, pd.DataFrame(all_results)], ignore_index=True
                )

        self.master_df = master_df

    def compute_bbox_and_area(self, rle_segmentation):
        """Computes bounding box and area from RLE segmentation.

        Args:
            rle_segmentation: Run-length encoded segmentation mask

        Returns:
            tuple: (bbox, area) where bbox is [x, y, width, height] and area is pixel count
        """
        if rle_segmentation:
            bbox = maskUtils.toBbox(rle_segmentation).tolist()
            area = maskUtils.area(rle_segmentation).tolist()
            return bbox, int(area)
        return [0, 0, 0, 0], 0  # Default values if segmentation is missing
