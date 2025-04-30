import json
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import altair as alt
import pandas as pd
from core_assist.dataset.converter import (convert_base, convert_coco,
                                           convert_yolo, seg_convert_base)
from core_assist.dataset.utils import (copyfile, filter_split_category,
                                       find_splits, ifnone, write_json)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split


class FormatSpec:
    """
    The base class to represent all annotation formats.

    This class provides functionality to handle different annotation formats for computer vision datasets.
    It supports operations like format conversion, dataset splitting, and visualization of data distributions.

    Args:
        root (Optional[Union[str, os.PathLike]]): Root directory containing the dataset
        has_split (Optional[bool]): Whether dataset is already split into train/val/test
        df (Optional[pd.DataFrame]): DataFrame containing annotations data
        format (Optional[str]): Format specification for the annotations
    """

    def __init__(
        self,
        root: Optional[Union[str, os.PathLike]] = None,
        has_split: Optional[bool] = False,
        df: Optional[pd.DataFrame] = None,
        format: Optional[str] = None,
    ):
        self.root = Path(root)
        self._has_image_split = has_split
        self.master_df = df
        self._format = format
        self._splits = None

    def format(self) -> str:
        """
        Get the format specification for this dataset.

        Returns:
            str: Format name derived from module name or specified format
        """
        if self._format is None:
            return self.__module__.split(".")[-1]
        return self._format

    def _resolve_dataframe(self):
        """
        Internal method to resolve and validate the DataFrame structure.
        To be implemented by child classes.
        """
        pass

    def __str__(self) -> str:
        """
        String representation showing format, root path and available splits.

        Returns:
            str: Formatted string with dataset information
        """
        return f"{self.format().upper()}[root:{self.root}, splits:[{', '.join(self._splits)}]]"

    def __repr__(self) -> str:
        """
        Official string representation of the format specification.

        Returns:
            str: Format name
        """
        return self._format()

    def describe(self) -> pd.DataFrame:
        """
        Shows basic data distribution across different splits.

        Aggregates statistics about number of images, annotations and categories per split.

        Returns:
            pd.DataFrame: DataFrame containing distribution statistics
        """
        df = (
            self.master_df.groupby(["split"])
            .agg(
                {"image_id": [pd.Series.nunique, "size"], "category": pd.Series.nunique}
            )
            .reset_index()
        )
        df.columns = (
            df.columns.get_level_values(0) + "_" + df.columns.get_level_values(1)
        )
        df.rename(
            columns={
                "image_id_nunique": "images",
                "image_id_size": "annotations",
                "category_nunique": "categories",
                "split_": "split",
            },
            inplace=True,
        )
        return df

    def bbox_stats(
        self, split: Optional[str] = None, category: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Computes bounding box descriptive statistics.

        Calculates statistics like mean, std dev etc. for bounding box dimensions.

        Args:
            split (Optional[str]): Dataset split to analyze (train/val/test)
            category (Optional[str]): Category to filter statistics for

        Returns:
            pd.DataFrame: DataFrame with bbox statistics
        """
        df = filter_split_category(self.master_df, split, category)
        return df[["x_min", "y_min", "width", "height"]].describe()

    def show_distribution(self) -> alt.Chart:
        """
        Plots distribution of labels across different splits.

        Creates a bar chart showing category distributions normalized within each split.

        Returns:
            alt.Chart: Altair chart object for visualization
        """
        df = self.master_df[["split", "category", "image_id"]].copy()
        distribution = (
            df.groupby(["split", "category"])["image_id"].size().rename("count")
        )
        distribution = pd.DataFrame(
            distribution / distribution.groupby(level=0).sum()
        ).reset_index()

        return (
            alt.Chart(distribution)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(x="category:O", y="count:Q", color="category", column="split")
        )

    def bbox_scatter(
        self,
        split: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 1000,
    ) -> alt.Chart:
        """
        Creates scatter plot of bounding box dimensions.

        Args:
            split (Optional[str]): Dataset split to visualize
            category (Optional[str]): Category to filter for
            limit (int): Maximum number of points to plot

        Returns:
            alt.Chart: Scatter plot of width vs height colored by category
        """
        df = filter_split_category(self.master_df, split, category).drop(
            "image_path", axis=1
        )
        limit = min(min(limit, len(df)), 5000)
        df = df.sample(n=limit, replace=False, random_state=42)
        return (
            alt.Chart(df)
            .mark_circle(size=30)
            .encode(x="width", y="height", color="category")
        )

    def _find_splits(self):
        """
        Internal method to detect dataset splits.

        Identifies train/val/test splits in the dataset directory structure.
        """
        splits, has_image_split = find_splits(
            self._image_dir, self._annotation_dir, self._format
        )
        self._has_image_split = has_image_split
        self._splits = splits

    def split(
        self, test_size: float = 0.2, stratified: bool = False, random_state: int = 42
    ):
        """
        Splits dataset into train and validation sets.

        Args:
            test_size (float): Fraction of data to use for validation
            stratified (bool): Whether to maintain class distribution in splits
            random_state (int): Random seed for reproducibility

        Returns:
            FormatSpec: New FormatSpec instance with split dataset
        """
        label_df = self.master_df.copy()

        if stratified:
            class_df = label_df[["image_id", "class_id"]].copy()
            class_df.drop_duplicates(inplace=True)
            gdf = (
                class_df.groupby("image_id")["class_id"]
                .agg(lambda x: x.tolist())
                .reset_index()
            )

            mlb = MultiLabelBinarizer()
            out = mlb.fit_transform(gdf.class_id)
            label_names = [f"class_{x}" for x in mlb.classes_]
            out = pd.DataFrame(data=out, columns=label_names)

            gdf = pd.concat([gdf, out], axis=1)
            gdf.drop(["class_id"], axis=1, inplace=True)

            train_images, _, test_images, _ = iterative_train_test_split(
                gdf[["image_id"]].values, gdf[label_names].values, test_size=test_size
            )

            train_images = train_images.ravel()
            test_images = test_images.ravel()

        else:
            image_ids = label_df.image_id.unique()
            train_images, test_images = train_test_split(
                image_ids, test_size=test_size, random_state=random_state
            )

        train_df = label_df.loc[label_df["image_id"].isin(train_images.tolist())]
        test_df = label_df.loc[label_df["image_id"].isin(test_images.tolist())]

        train_df.loc[:, "split"] = "train"
        test_df.loc[:, "split"] = "valid"

        master_df = pd.concat([train_df, test_df], ignore_index=True)
        return FormatSpec(self.root, True, master_df, format=self.format)

    def save(
        self,
        output_dir: Optional[Union[str, os.PathLike]],
        export_to: Optional[str] = None,
        copy_images: bool = True,
    ):
        """
        Saves dataset in specified format.

        Args:
            output_dir: Directory to save converted dataset
            export_to: Target format to convert to
            copy_images: Whether to copy image files

        Returns:
            Result of conversion operation
        """
        export_to = ifnone(export_to, self.format)
        return self.convert(export_to, output_dir=output_dir, copy_images=copy_images)

    def convert(
        self,
        to: str,
        output_dir: Optional[str] = None,
        save_under: Optional[str] = None,
        copy_images: bool = False,
        **kwargs,
    ):
        """
        Converts dataset to specified format.

        Args:
            to: Target format to convert to
            output_dir: Directory to save converted dataset
            save_under: Subdirectory to save under
            copy_images: Whether to copy image files
            **kwargs: Additional format-specific arguments

        Returns:
            Result of conversion operation

        Raises:
            NotImplementedError: If target format is not supported
        """
        if to.lower() == "yolo":
            return convert_yolo(
                self.master_df,
                self.root,
                copy_images=copy_images,
                save_under=save_under,
                output_dir=output_dir,
            )
        elif to.lower() == "coco":
            return convert_coco(
                self.master_df,
                self.root,
                copy_images=copy_images,
                save_under=save_under,
                output_dir=output_dir,
            )
        elif to.lower() == "base":
            return convert_base(
                self.master_df,
                self.root,
                output_dir=output_dir,
                save_under=save_under,
                copy_images=copy_images,
            )
        else:
            raise NotImplementedError


import json
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from core_assist.dataset.converter import (seg_convert_base, seg_convert_coco,
                                           seg_convert_detectron,
                                           seg_convert_mask, seg_convert_yolo)
from core_assist.dataset.utils import copyfile, find_splits, ifnone, write_json


class SegFormatSpec:
    """
    Base class for segmentation annotation formats.

    Handles segmentation mask annotations in various formats like COCO, YOLO etc.
    Provides functionality for format conversion and dataset analysis.

    Args:
        root: Root directory containing dataset
        has_split: Whether dataset is already split
        df: DataFrame containing annotations
        format: Format specification
    """

    def __init__(
        self,
        root: Optional[Union[str, os.PathLike]] = None,
        has_split: Optional[bool] = False,
        df: Optional[pd.DataFrame] = None,
        format: Optional[str] = None,
    ):
        self.root = Path(root)
        self._has_image_split = has_split
        self.master_df = df
        self._format = format
        self._splits = None

    def format(self):
        """
        Get format specification.

        Returns:
            str: Format name
        """
        if self._format is None:
            return self.__module__.split(".")[-1]
        return self._format

    def _resolve_dataframe(self):
        """
        Internal method to resolve DataFrame structure.
        To be implemented by child classes.
        """
        pass

    def __str__(self):
        """
        String representation with format and splits info.

        Returns:
            str: Formatted string with dataset info
        """
        return f"{self.format().upper()}[root:{self.root}, splits:[{', '.join(self._splits)}]]"

    def __repr__(self):
        """
        Official string representation.

        Returns:
            str: Format name
        """
        return self._format()

    def show_distribution(self) -> alt.Chart:
        """
        Plot label distribution across splits.

        Returns:
            alt.Chart: Bar chart showing category distributions
        """
        df = self.master_df[["split", "category", "image_id"]].copy()
        distribution = (
            df.groupby(["split", "category"])["image_id"].size().rename("count")
        )
        distribution = pd.DataFrame(
            distribution / distribution.groupby(level=0).sum()
        ).reset_index()

        return (
            alt.Chart(distribution)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(x="category:O", y="count:Q", color="category", column="split")
        )

    def describe(self) -> pd.DataFrame:
        """
        Show basic data distribution per split.

        Returns:
            pd.DataFrame: Statistics about images, annotations and categories
        """
        df = (
            self.master_df.groupby(["split"])
            .agg(
                {"image_id": [pd.Series.nunique, "size"], "category": pd.Series.nunique}
            )
            .reset_index()
        )
        df.columns = (
            df.columns.get_level_values(0) + "_" + df.columns.get_level_values(1)
        )
        df.rename(
            columns={
                "image_id_nunique": "images",
                "image_id_size": "annotations",
                "category_nunique": "categories",
                "split_": "split",
            },
            inplace=True,
        )
        return df

    def bbox_stats(
        self, split: Optional[str] = None, category: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute bounding box statistics.

        Args:
            split: Dataset split to analyze
            category: Category to filter for

        Returns:
            pd.DataFrame: Statistics of bounding boxes
        """
        df = filter_split_category(self.master_df, split, category)
        return df[["x_min", "y_min", "width", "height"]].describe()

    def bbox_scatter(
        self,
        split: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 1000,
    ) -> alt.Chart:
        """
        Create scatter plot of bbox dimensions.

        Args:
            split: Dataset split to visualize
            category: Category to filter for
            limit: Maximum points to plot

        Returns:
            alt.Chart: Scatter plot of width vs height
        """
        df = filter_split_category(self.master_df, split, category).drop(
            "image_path", axis=1
        )
        limit = min(min(limit, len(df)), 5000)
        df = df.sample(n=limit, replace=False, random_state=42)
        return (
            alt.Chart(df)
            .mark_circle(size=30)
            .encode(x="width", y="height", color="category")
        )

    def _find_splits(self):
        """
        Internal method to detect dataset splits.
        """
        splits, has_image_split = find_splits(
            self._image_dir, self._annotation_dir, self._format
        )
        self._has_image_split = has_image_split
        self._splits = splits

    def split(
        self, test_size: float = 0.2, stratified: bool = False, random_state: int = 42
    ):
        """
        Split dataset into train and validation sets.

        Args:
            test_size: Fraction for validation
            stratified: Whether to maintain class distribution
            random_state: Random seed

        Returns:
            SegFormatSpec: New instance with split dataset
        """
        label_df = self.master_df.copy()

        if stratified:
            class_df = label_df[["image_id", "class_id"]].copy()
            class_df.drop_duplicates(inplace=True)
            gdf = (
                class_df.groupby("image_id")["class_id"]
                .agg(lambda x: x.tolist())
                .reset_index()
            )

            mlb = MultiLabelBinarizer()
            out = mlb.fit_transform(gdf.class_id)
            label_names = [f"class_{x}" for x in mlb.classes_]
            out = pd.DataFrame(data=out, columns=label_names)

            gdf = pd.concat([gdf, out], axis=1)
            gdf.drop(["class_id"], axis=1, inplace=True)

            train_images, _, test_images, _ = iterative_train_test_split(
                gdf[["image_id"]].values, gdf[label_names].values, test_size=test_size
            )

            train_images = train_images.ravel()
            test_images = test_images.ravel()

        else:
            image_ids = label_df.image_id.unique()
            train_images, test_images = train_test_split(
                image_ids, test_size=test_size, random_state=random_state
            )

        train_df = label_df.loc[label_df["image_id"].isin(train_images.tolist())]
        test_df = label_df.loc[label_df["image_id"].isin(test_images.tolist())]

        train_df.loc[:, "split"] = "train"
        test_df.loc[:, "split"] = "valid"

        master_df = pd.concat([train_df, test_df], ignore_index=True)
        return SegFormatSpec(self.root, True, master_df, format=self.format)

    def save(
        self,
        output_dir: Optional[Union[str, os.PathLike]],
        export_to: Optional[str] = None,
        copy_images: bool = True,
    ):
        """
        Save dataset in specified format.

        Args:
            output_dir: Directory to save converted dataset
            export_to: Target format
            copy_images: Whether to copy images

        Returns:
            Result of conversion operation
        """
        export_to = ifnone(export_to, self.format)
        return self.convert(export_to, output_dir=output_dir, copy_images=copy_images)

    def convert(
        self,
        to: str,
        output_dir: Optional[str] = None,
        save_under: Optional[str] = None,
        copy_images: bool = False,
        **kwargs,
    ):
        """
        Convert dataset to specified format.

        Args:
            to: Target format
            output_dir: Output directory
            save_under: Subdirectory
            copy_images: Whether to copy images
            **kwargs: Additional arguments

        Returns:
            Result of conversion operation

        Raises:
            NotImplementedError: If format not supported
        """
        if to.lower() == "yolo":
            return seg_convert_yolo(
                self.master_df,
                self.root,
                copy_images=copy_images,
                save_under=save_under,
                output_dir=output_dir,
            )
        elif to.lower() == "coco":
            return seg_convert_coco(
                self.master_df,
                self.root,
                copy_images=copy_images,
                save_under=save_under,
                output_dir=output_dir,
            )
        elif to.lower() == "mask":
            return seg_convert_mask(
                self.master_df,
                self.root,
                copy_images=copy_images,
                save_under=save_under,
                output_dir=output_dir,
            )
        elif to.lower() == "detectron2":
            return seg_convert_detectron(
                self.master_df,
                self.root,
                copy_images=copy_images,
                save_under=save_under,
                output_dir=output_dir,
            )
        elif to.lower() == "base":
            return seg_convert_base(
                self.master_df,
                self.root,
                output_dir=output_dir,
                save_under=save_under,
                copy_images=copy_images,
            )
        else:
            raise NotImplementedError
