import hashlib
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


def extension(file_paths, valid_formats=[".jpg", ".png", ".jpeg"]):
    """
    Validate that the file has a valid image format.
    """
    res = []
    try:
        for file_path in file_paths:
            _, ext = os.path.splitext(file_path)
            res.append(ext.lower() in valid_formats)

        return res
    except Exception as e:
        print(f"Error validating file extension: {e}")
        return False


def path(file_path):
    """
    Validate if the file path exists and is a file.
    """
    try:
        return os.path.isfile(file_path)
    except Exception as e:
        print(f"Error validating file path: {e}")
        return False


def image(file_path):
    """
    Check if the image is corrupt or invalid.
    """
    try:
        image = cv2.imread(file_path)
        return image is not None
    except Exception as e:
        print(f"Error validating image: {e}")
        return False


def hash_file(file_path):
    """
    Generate a hash for the image file to detect duplicates.
    """
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Error generating hash: {e}")
        return None


def find_content_duplicate(dir):
    """
    Detect duplicate images in the directory using hash comparison.
    Returns a list of duplicate file names.
    """
    try:
        hashes = {}
        duplicates = []
        for file_name in os.listdir(dir):
            file_path = os.path.join(dir, file_name)
            if os.path.isfile(file_path):
                file_hash = hash_file(file_path)
                if file_hash is None:
                    continue
                if file_hash in hashes:
                    duplicates.append(file_name)  # Append the duplicate file name
                else:
                    hashes[file_hash] = (
                        file_name  # Store the hash and its corresponding file
                    )
        return duplicates
    except Exception as e:
        print(f"Error finding content duplicates: {e}")
        return []


def find_dir_common_files(dir1, dir2):
    """
    Find common file names between two directories.

    :param dir1: Path to the first directory.
    :param dir2: Path to the second directory.
    :return: List of common file names.
    """
    try:
        if not os.path.isdir(dir1) or not os.path.isdir(dir2):
            raise ValueError("One or both directories do not exist or are invalid.")

        # Get list of file names in both directories
        files_in_dir1 = set(os.listdir(dir1))
        files_in_dir2 = set(os.listdir(dir2))

        # Find common files
        common_files = list(files_in_dir1.intersection(files_in_dir2))

        return common_files

    except Exception as e:
        print(f"Error finding common files: {e}")
        return None


def image_size_and_dim(file_path, min_dim=(50, 50), max_size_mb=5):
    """
    Check if the image meets the required dimensions and file size without fully loading the image.
    """
    try:
        with Image.open(file_path) as image:
            width, height = image.size
        size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Get file size in MB
        return width >= min_dim[0] and height >= min_dim[1] and size_mb <= max_size_mb
    except UnidentifiedImageError:
        print(f"Error: File at {file_path} is not a valid image.")
    except Exception as e:
        print(f"Error checking image dimensions and size: {e}")
    return False


def match_mask_size(image_path, mask_path):
    """
    Check if the mask size matches the image size for segmentation without fully loading the image and mask.
    """
    try:
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image.verify()
            mask.verify()
            return image.size == mask.size
    except UnidentifiedImageError:
        print(f"Error: Either {image_path} or {mask_path} is not a valid image.")
    except Exception as e:
        print(f"Error matching mask size: {e}")
    return False


def search_dir(dir, filename):
    """
    Check if a file exists in the specified directory.
    """
    try:
        return os.path.isfile(os.path.join(dir, filename))
    except Exception as e:
        print(f"Error searching for file: {e}")
        return False


def search_csv(csv_path, column, to_search):
    """
    Check if a value exists in a specific column of a CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
        return (
            to_search in df[column].values
        )  # Check if value exists in specified column
    except FileNotFoundError:
        print(f"Error: CSV file at {csv_path} not found.")
    except KeyError:
        print(f"Error: Column '{column}' does not exist in the CSV file.")
    except Exception as e:
        print(f"Error searching CSV: {e}")
    return False


# def rectify_csv(df, img_paths):
#     """
#     Remove rows from the DataFrame where the 'img_path' column matches any path in the given list of image paths.

#     :param df: pandas.DataFrame - The input DataFrame containing a column named 'img_path'.
#     :param img_paths: list - A list of image paths to be removed from the DataFrame.
#     :return: pandas.DataFrame - A new DataFrame with the specified rows removed.
#     """
#     if "img_path" not in df.columns:
#         raise ValueError("The DataFrame must contain a column named 'img_path'.")

#     # Filter the DataFrame to exclude rows where 'img_path' matches any path in img_paths
#     filtered_df = df[~df["img_path"].isin(img_paths)]
#     return filtered_df
