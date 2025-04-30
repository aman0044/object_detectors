import hashlib
import os
import shutil
import tempfile

import cv2
import numpy as np
import pandas as pd
import pytest
from core_assist.validate.validate import (extension, find_content_duplicate,
                                           find_name_duplicate, hash_file,
                                           image, image_size_and_dim,
                                           match_mask_size, path, search_csv,
                                           search_dir)
from PIL import Image


# Temporary directory and files setup
@pytest.fixture
def temp_directory():
    """Fixture to create a temporary directory and files."""
    temp_dir = tempfile.mkdtemp()

    # Create some test files
    with open(os.path.join(temp_dir, "image1.jpg"), "w") as f:
        f.write("Test Image Content")
    with open(os.path.join(temp_dir, "image2.jpg"), "w") as f:
        f.write("Test Image Content")
    with open(os.path.join(temp_dir, "image3.png"), "w") as f:
        f.write("Test Image Content")

    # Create a sample CSV
    csv_content = "filename,column\nimage1.jpg,abc\nimage2.jpg,def"
    with open(os.path.join(temp_dir, "file_list.csv"), "w") as f:
        f.write(csv_content)

    yield temp_dir  # Return directory to tests

    shutil.rmtree(temp_dir)  # Cleanup after tests


# Test for extension function
def test_extension(temp_directory):
    assert extension(os.path.join(temp_directory, "image1.jpg"))
    assert not extension(os.path.join(temp_directory, "image1.txt"))


# Test for path function
def test_path(temp_directory):
    assert path(os.path.join(temp_directory, "image1.jpg"))
    assert not path(os.path.join(temp_directory, "non_existing_file.jpg"))


# Test for image function (fake a corrupt image)
def test_image(temp_directory):
    # Create a real valid image file
    img = np.zeros((10, 10), dtype=np.uint8)
    cv2.imwrite(os.path.join(temp_directory, "valid_image.jpg"), img)

    assert image(os.path.join(temp_directory, "valid_image.jpg"))
    assert not image(os.path.join(temp_directory, "invalid_image.txt"))


# Test for hash_file function
def test_hash_file(temp_directory):
    file_path = os.path.join(temp_directory, "image1.jpg")
    hash_value = hash_file(file_path)
    assert isinstance(hash_value, str)
    assert len(hash_value) == 32  # MD5 hash length


# # Test for find_content_duplicate function
# def test_find_content_duplicate(temp_directory):
#     duplicate_files = find_content_duplicate(temp_directory)
#     assert 'image2.jpg' in duplicate_files

# # Test for find_name_duplicate function
# def test_find_name_duplicate(temp_directory):
#     duplicate_files = find_name_duplicate(temp_directory)
#     assert 'image2.jpg' in duplicate_files


# Test for image_size_and_dim function
def test_image_size_and_dim(temp_directory):
    # Create a 100x100 image file (valid size)
    img = Image.new("RGB", (100, 100))
    img.save(os.path.join(temp_directory, "valid_image.jpg"))

    assert image_size_and_dim(
        os.path.join(temp_directory, "valid_image.jpg"), min_dim=(50, 50), max_size_mb=5
    )
    assert not image_size_and_dim(
        os.path.join(temp_directory, "valid_image.jpg"),
        min_dim=(200, 200),
        max_size_mb=5,
    )


# Test for match_mask_size function
def test_match_mask_size(temp_directory):
    # Create a valid image
    img = Image.new("RGB", (100, 100))
    img.save(os.path.join(temp_directory, "image.jpg"))

    # Create a mask with the same size
    mask = Image.new("1", (100, 100))
    mask.save(os.path.join(temp_directory, "mask.jpg"))

    # Create a mask with different size
    mask_diff = Image.new("1", (200, 200))
    mask_diff.save(os.path.join(temp_directory, "mask_diff.jpg"))

    assert match_mask_size(
        os.path.join(temp_directory, "image.jpg"),
        os.path.join(temp_directory, "mask.jpg"),
    )
    assert not match_mask_size(
        os.path.join(temp_directory, "image.jpg"),
        os.path.join(temp_directory, "mask_diff.jpg"),
    )


# Test for search_dir function
def test_search_dir(temp_directory):
    assert search_dir(temp_directory, "image1.jpg")
    assert not search_dir(temp_directory, "non_existing_file.jpg")


# Test for search_csv function
def test_search_csv(temp_directory):
    assert search_csv(
        os.path.join(temp_directory, "file_list.csv"), "filename", "image1.jpg"
    )
    assert not search_csv(
        os.path.join(temp_directory, "file_list.csv"),
        "filename",
        "non_existing_file.jpg",
    )
