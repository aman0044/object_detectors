import cv2
import numpy as np
import pytest
from core_assist.plot.plot import (bbox, display_multiple_images,
                                   generate_color_for_text, image, segment,
                                   segments)
from matplotlib import pyplot as plt


@pytest.fixture
def sample_image():
    """Fixture to create a sample image for testing."""
    # Create a simple image (a black square with a white rectangle)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)  # White rectangle
    return img


@pytest.fixture
def sample_masks():
    """Fixture to create sample binary masks for testing."""
    # Create binary mask of a white rectangle
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 1, -1)
    return [mask]  # List of one mask


@pytest.fixture
def sample_bboxes():
    """Fixture to create sample bounding boxes for testing."""
    # Create bounding box (x, y, width, height)
    return [(20, 20, 60, 60)]  # A single bounding box


@pytest.fixture
def sample_labels():
    """Fixture to create sample labels for testing."""
    return ["Object"]  # Sample label for bounding box and mask


def test_generate_color_for_text():
    """Test the consistent color generation for text."""
    text1 = "Object1"
    text2 = "Object2"

    color1 = generate_color_for_text(text1)
    color2 = generate_color_for_text(text2)

    # Test that the colors are consistent for the same text
    assert generate_color_for_text(text1) == color1
    assert generate_color_for_text(text2) == color2
    assert color1 != color2  # Different text should generate different colors


def test_image_plot(sample_image):
    """Test the image function to display a single image."""
    image(img=sample_image)


def test_display_multiple_images(sample_image):
    """Test the display_multiple_images function."""
    display_multiple_images(
        [sample_image, sample_image], labels=["Image 1", "Image 2"], rows=1, cols=2
    )


def test_bbox_plot(sample_image, sample_bboxes, sample_labels):
    """Test the bbox function with a sample image."""
    bbox(sample_image, sample_bboxes, labels=sample_labels)


def test_segment_plot(sample_image, sample_masks, sample_labels, sample_bboxes):
    """Test the segment function to overlay masks and bounding boxes."""
    segment(
        sample_image,
        sample_masks,
        sample_labels,
        bboxes=sample_bboxes,
        bbox_labels=sample_labels,
    )


def test_segments_plot(sample_image, sample_masks, sample_labels, sample_bboxes):
    """Test the segments function with multiple images, masks, and bounding boxes."""
    segments(
        [sample_image],
        [sample_masks],
        [sample_labels],
        bboxes=[sample_bboxes],
        bbox_labels=[sample_labels],
    )
