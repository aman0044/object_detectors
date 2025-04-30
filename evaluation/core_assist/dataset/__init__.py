import os
import random
import warnings
from pathlib import Path
from typing import Optional, Union

from core_assist.dataset.base import Base, SegmentationBase
from core_assist.dataset.coco import Coco, SegmentationCoco
from core_assist.dataset.detectron import SegmentationDetectron
from core_assist.dataset.format import FormatSpec
from core_assist.dataset.mask import SegmentationMask
from core_assist.dataset.seg_visual import SegVisualizer
from core_assist.dataset.utils import get_image_dir, ifnone
from core_assist.dataset.visual import Visualizer
from core_assist.dataset.yolo import SegmentationYolo, Yolo

SUPPORTED_FORMATS = {
    "coco": Coco,
    "base": Base,
    "yolo": Yolo,
}


class DetDataset:
    """A class for handling object detection datasets in various formats.

    This class provides a unified interface for working with object detection datasets
    in different annotation formats (COCO, YOLO, etc). It supports operations like:
    - Loading and parsing annotations
    - Dataset visualization
    - Format conversion
    - Train/test splitting
    - Statistics and analysis

    Args:
        root (Union[str, Path]): Root directory containing the dataset
        format (str): Format of the annotations ('coco', 'yolo', 'base')
        mapping (Optional[dict]): Class ID to name mapping
        is_csv (bool): Whether annotations are in CSV format

    Attributes:
        root: Root directory path
        format: Annotation format
        mapping: Class mapping dictionary
        formatspec: Format-specific handler instance
    """

    def __init__(self, root, format, mapping=None, is_csv=False):
        self.root = root
        self.format = format.lower()  # Convert to lowercase for consistency
        self.mapping = mapping
        self.is_csv = False
        assert self.format in SUPPORTED_FORMATS, f"Unsupported format: {self.format}"
        self.formatspec = SUPPORTED_FORMATS[self.format](
            root, format=self.format, mapping=self.mapping
        )

    def __str__(self):
        return self.formatspec.__str__()

    def __repr__(self):
        return self.formatspec.__repr__()

    @property
    def splits(self):
        """Get dataset splits (train/val/test)"""
        return self.formatspec.splits

    @property
    def label_df(self):
        """Get annotation data as pandas DataFrame"""
        return self.formatspec.master_df

    @property
    def describe(self):
        """Get dataset statistics and description"""
        return self.formatspec.describe()

    def show_distribution(self):
        """Visualize class distribution in dataset"""
        return self.formatspec.show_distribution()

    def bbox_scatter(self, split: Optional[str] = None, category: Optional[str] = None):
        """Plot bounding box scatter visualization

        Args:
            split: Dataset split to visualize
            category: Category to filter by
        """
        return self.formatspec.bbox_scatter(split, category)

    def bbox_stats(self, split: Optional[str] = None, category: Optional[str] = None):
        """Get bounding box statistics

        Args:
            split: Dataset split to analyze
            category: Category to filter by
        """
        return self.formatspec.bbox_stats(split, category)

    def export(
        self, to: str, output_dir: Optional[Union[str, os.PathLike]] = None, **kwargs
    ):
        """Export dataset to a different format

        Args:
            to: Target format
            output_dir: Output directory path
            **kwargs: Additional export options
        """
        if not to.lower() in SUPPORTED_FORMATS:
            raise ValueError(f"`{to}` is not a supported conversion format")

        return self.formatspec.convert(to.lower(), output_dir=output_dir, **kwargs)

    def train_test_split(
        self, test_size: float = 0.2, stratified: bool = False, random_state: int = 42
    ):
        """Split dataset into train and validation sets

        Args:
            test_size: Fraction of data for validation
            stratified: Whether to maintain class distribution
            random_state: Random seed for reproducibility

        Returns:
            FormatSpec: New FormatSpec with split dataset
        """
        return self.formatspec.split(test_size, stratified, random_state)

    def visualizer(
        self,
        image_dir: Optional[Union[str, os.PathLike]] = None,
        split: Optional[str] = None,
        img_size: Optional[int] = 512,
        **kwargs,
    ):
        """Create visualization interface for the dataset

        Args:
            image_dir: Directory containing images
            split: Dataset split to visualize
            img_size: Size to resize images to
            **kwargs: Additional visualization options

        Returns:
            Visualizer: Visualization interface instance
        """
        if image_dir is None:
            random_split = random.choice(list(self.formatspec.master_df.split.unique()))
            if split is None:
                split = random_split
                warnings.warn(
                    f"Since there is not split specified explicitly, {split} has been selected randomly."
                    + "Please specify split if you want to visualize different split."
                )
            if self.formatspec._has_image_split:
                image_dir = get_image_dir(self.root) / split
            else:
                image_dir = get_image_dir(self.root)
        image_dir = Path(image_dir)
        return Visualizer(
            image_dir, self.formatspec.master_df, split, img_size, **kwargs
        )


SSUPPORTED_FORMATS = {
    "coco": SegmentationCoco,
    "base": SegmentationBase,
    "yolo": SegmentationYolo,
    "mask": SegmentationMask,
    "detectron2": SegmentationDetectron,
}


class SegDataset:
    """A class for handling segmentation datasets in various formats.

    This class provides a unified interface for working with segmentation datasets
    in different annotation formats (COCO, YOLO, Mask, etc). It supports operations like:
    - Loading and parsing segmentation masks/annotations
    - Dataset visualization
    - Format conversion
    - Train/test splitting
    - Statistics and analysis

    Args:
        root (Union[str, Path]): Root directory containing the dataset
        format (str): Format of the annotations ('coco', 'yolo', 'mask', etc)
        mapping (Optional[dict]): Class ID to name mapping
        is_csv (bool): Whether annotations are in CSV format

    Attributes:
        root: Root directory path
        format: Annotation format
        mapping: Class mapping dictionary
        formatspec: Format-specific handler instance
    """

    def __init__(self, root, format, mapping=None, is_csv=False):
        self.root = root
        self.format = format.lower()  # Convert to lowercase for consistency
        self.mapping = mapping
        self.is_csv = False
        assert self.format in SSUPPORTED_FORMATS, f"Unsupported format: {self.format}"
        self.formatspec = SSUPPORTED_FORMATS[self.format](
            root, format=self.format, mapping=self.mapping
        )

    def __str__(self):
        return self.formatspec.__str__()

    def __repr__(self):
        return self.formatspec.__repr__()

    @property
    def splits(self):
        """Get dataset splits (train/val/test)"""
        return self.formatspec.splits

    @property
    def label_df(self):
        """Get annotation data as pandas DataFrame"""
        return self.formatspec.master_df

    def show_distribution(self):
        """Visualize class distribution in dataset"""
        return self.formatspec.show_distribution()

    def bbox_scatter(self, split: Optional[str] = None, category: Optional[str] = None):
        """Plot bounding box scatter visualization

        Args:
            split: Dataset split to visualize
            category: Category to filter by
        """
        return self.formatspec.bbox_scatter(split, category)

    def bbox_stats(self, split: Optional[str] = None, category: Optional[str] = None):
        """Get bounding box statistics

        Args:
            split: Dataset split to analyze
            category: Category to filter by
        """
        return self.formatspec.bbox_stats(split, category)

    @property
    def describe(self):
        """Get dataset statistics and description"""
        return self.formatspec.describe()

    def export(
        self, to: str, output_dir: Optional[Union[str, os.PathLike]] = None, **kwargs
    ):
        """Export dataset to a different format

        Args:
            to: Target format
            output_dir: Output directory path
            **kwargs: Additional export options
        """
        if not to.lower() in SSUPPORTED_FORMATS:
            raise ValueError(f"`{to}` is not a supported conversion format")

        return self.formatspec.convert(to.lower(), output_dir=output_dir, **kwargs)

    def train_test_split(
        self, test_size: float = 0.2, stratified: bool = False, random_state: int = 42
    ):
        """Split dataset into train and validation sets

        Args:
            test_size: Fraction of data for validation
            stratified: Whether to maintain class distribution
            random_state: Random seed for reproducibility

        Returns:
            FormatSpec: New FormatSpec with split dataset
        """
        return self.formatspec.split(test_size, stratified, random_state)

    def visualizer(
        self,
        image_dir: Optional[Union[str, os.PathLike]] = None,
        split: Optional[str] = None,
        img_size: Optional[int] = 512,
        **kwargs,
    ):
        """Create visualization interface for the segmentation dataset

        Args:
            image_dir: Directory containing images
            split: Dataset split to visualize
            img_size: Size to resize images to
            **kwargs: Additional visualization options

        Returns:
            SegVisualizer: Segmentation visualization interface instance
        """
        if image_dir is None:
            random_split = random.choice(list(self.formatspec.master_df.split.unique()))
            if split is None:
                split = random_split
                warnings.warn(
                    f"Since there is not split specified explicitly, {split} has been selected randomly."
                    + "Please specify split if you want to visualize different split."
                )
            if self.formatspec._has_image_split:
                image_dir = get_image_dir(self.root) / split
            else:
                image_dir = get_image_dir(self.root)
        image_dir = Path(image_dir)
        return SegVisualizer(
            image_dir, self.formatspec.master_df, split, img_size, **kwargs
        )
