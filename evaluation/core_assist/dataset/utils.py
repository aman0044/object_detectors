import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd


def ifnone(
    x: Any, y: Any, transform: Optional[Callable] = None, type_safe: bool = False
):
    """if x is None return y otherwise x after applying transofrmation ``transform`` and
    casting the result back to original type if ``type_safe``

    Args:
        x (Any): returns x if x is not none
        y (Any): returns y if x is none
        transform (Optional[Callable], optional): applies transform to the output. Defaults to None.
        type_safe (bool, optional): if true, tries casting the output to the original type. Defaults to False.
    """

    if transform is not None:
        assert callable(
            transform
        ), "`transform` should be either `None` or instance of `Callable`"
    else:

        def transform(x):
            return x

    if x is None:
        orig_type = type(y)
        out = transform(y)
    else:
        orig_type = type(x)
        out = transform(x)
    if type_safe:
        try:
            out = orig_type(out)
        except (ValueError, TypeError):
            warnings.warn(f"output could not be casted as type {orig_type.__name__}")
            pass
    return out


def find_splits(
    image_dir: Union[str, os.PathLike],
    annotation_dir: Union[str, os.PathLike],
    format: str,
):
    """find the splits in the dataset, will ignore splits for which no annotation is found"""

    # print(f"passed format: {format}")

    exts = {"coco": "json", "base": "json", "yolo": "txt"}

    ext = exts[format]

    im_splits = [
        x.name
        for x in Path(image_dir).iterdir()
        if x.is_dir() and not x.name.startswith(".")
    ]

    if format in ("yolo", "pascal"):
        ann_splits = [x.name for x in Path(annotation_dir).iterdir() if x.is_dir()]

        if not ann_splits:
            files = list(Path(annotation_dir).glob(f"*.{ext}"))
            if len(files):
                ann_splits = ["main"]
            else:
                raise ValueError(
                    "No annotation found. Please check the directory specified."
                )

    else:
        ann_splits = [x.stem for x in Path(annotation_dir).glob(f"*.{ext}")]

    no_anns = set(im_splits).difference(ann_splits)
    if no_anns:
        warnings.warn(f"no annotation found for {', '.join(list(no_anns))}")

    return ann_splits, len(im_splits) > 0


def read_coco(coco_json: Union[str, os.PathLike]):
    """read a coco json and returns the images, annotations and categories dict separately"""
    with open(coco_json, "r") as f:
        coco = json.load(f)
    return coco["images"], coco["annotations"], coco["categories"]


def write_json(data_dict: Dict, filename: Union[str, os.PathLike]):
    """writes json to disk"""
    with open(filename, "w") as f:
        json.dump(data_dict, f, indent=2)


def copyfile(
    src: Union[str, os.PathLike],
    dest: Union[str, os.PathLike],
    filename: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """copies a file from one path to another

    Args:
        src (Union[str, os.PathLike]): either a directory containing files or any filepath.
        dest (Union[str, os.PathLike]): the output directory for the copy.
        filename (Optional[Union[str, os.PathLike]], optional): If ``src`` is a directory, the name of the
           file to copy. Defaults to None.
    """
    if filename is not None:
        filename = Path(src) / filename

    else:
        filename = Path(src)

    dest = Path(dest) / filename.name
    try:
        shutil.copyfile(filename, dest)
    except FileNotFoundError:
        pass


def exists(path: Union[str, os.PathLike]):
    """checks for whether a directory or file exists in the specified path"""
    if Path(path).is_dir():
        return "dir"

    if Path(path).is_file():
        return "file"

    return


def get_image_dir(root: Union[str, os.PathLike]):
    """returns image directory given a root directory"""
    return Path(root) / "images"


def get_annotation_dir(root: Union[str, os.PathLike]):
    """returns annotation directory given a root directory"""
    return Path(root) / "annotations"


def find_job_metadata_key(json_data: Dict):
    """finds metadata key for sagemaker manifest format"""
    for key in json_data.keys():
        if key.split("-")[-1] == "metadata":
            return key


def read_coco(coco_json: Union[str, os.PathLike]):
    """read a coco json and returns the images, annotations and categories dict separately"""
    with open(coco_json, "r") as f:
        coco = json.load(f)
    return coco["images"], coco["annotations"], coco["categories"]


def write_json(data_dict: Dict, filename: Union[str, os.PathLike]):
    """writes json to disk"""
    with open(filename, "w") as f:
        json.dump(data_dict, f, indent=2)


def filter_split_category(
    df: pd.DataFrame, split: Optional[str] = None, category: Optional[str] = None
) -> pd.DataFrame:
    """given the label df, filters the dataframe by split and/or label category

    Args:
        df (pd.DataFrame): the label dataframe.
        split (Optional[str], optional): the dataset split e.g., ``train``, ``test`` etc. Defaults to None.
        category (Optional[str], optional): the label category. Defaults to None.

    Raises:
        ValueError: if an unknown category is specified.

    Returns:
        pd.DataFrame: the filtered dataframe.
    """
    if split is not None:
        df = df.query("split == @split").copy()

    if category is not None:
        if category not in df.category.unique():
            raise ValueError(f"class `{category}` is not present in annotations")
        df = df.query("category == @category").copy()

    return df


def copyfile(
    src: Union[str, os.PathLike],
    dest: Union[str, os.PathLike],
    filename: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """copies a file from one path to another

    Args:
        src (Union[str, os.PathLike]): either a directory containing files or any filepath.
        dest (Union[str, os.PathLike]): the output directory for the copy.
        filename (Optional[Union[str, os.PathLike]], optional): If ``src`` is a directory, the name of the
           file to copy. Defaults to None.
    """
    if filename is not None:
        filename = Path(src) / filename

    else:
        filename = Path(src)

    dest = Path(dest) / filename.name
    try:
        shutil.copyfile(filename, dest)
    except FileNotFoundError:
        pass
