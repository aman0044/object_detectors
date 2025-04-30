import os
from io import BytesIO
from typing import Optional, Tuple, Union

import cv2
import numpy as np


# Load image
def load_img(img_path):
    """
    Loads an image from the specified path using OpenCV.
    """
    try:
        # Open the image using OpenCV (cv2.imread returns BGR by default)
        img = cv2.imread(img_path)
        if img is None:
            raise Exception("Failed to load image")
        return img
    except Exception as e:
        print(f"Failed to load image from path: {img_path}")
        print("Error:", e)
        return None


# Load RGB image
def load_rgb(img_path):
    """
    Loads an image and ensures it's in RGB format using OpenCV.
    """
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        return None

    img = cv2.imread(img_path)  # OpenCV loads images in BGR format
    if img is None:
        print(f"Error: Failed to load image from {img_path}")
        return None

    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img
    except Exception as e:
        print(f"Error converting image to RGB: {e}")
        return None


# Load BGR image
def load_bgr(img_path):
    """
    Loads an image and returns it in BGR format (OpenCV default).
    """
    img = load_img(img_path)
    try:
        if img is not None:
            # Return the BGR image (OpenCV default format)
            return img
    except Exception as e:
        print("Error:", e)
        return None


# Load Gray image
def load_gray(img_path):
    """
    Loads an image and converts it to grayscale using OpenCV.
    """
    img = load_img(img_path)
    try:
        if img is not None:
            # Convert to grayscale using OpenCV
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return gray_img
    except Exception as e:
        print("Error:", e)
        return None


# Load RGB image from buffer
def load_buffer_rgb(buffer_data):
    """
    Loads an image from buffer data and converts it to RGB mode.
    """
    try:
        # Read the image from buffer data (NumPy array)
        img_array = np.asarray(bytearray(buffer_data), dtype=np.uint8)
        img = cv2.imdecode(
            img_array, cv2.IMREAD_COLOR
        )  # Decode the image as color (BGR)

        if img is not None:
            # Convert from BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return rgb_img
    except Exception as e:
        print("Error:", e)
        return None


# Load BGR image from buffer
def load_buffer_bgr(buffer_data):
    """
    Loads an image from buffer data and converts it to BGR mode.
    """
    try:
        # Read the image from buffer data (NumPy array)
        img_array = np.asarray(bytearray(buffer_data), dtype=np.uint8)
        img = cv2.imdecode(
            img_array, cv2.IMREAD_COLOR
        )  # Decode the image as color (BGR)

        return img
    except Exception as e:
        print("Error:", e)
        return None


# Load gray image from buffer
def load_buffer_gray(buffer_data):
    """
    Loads an image from buffer data and converts it to grayscale.
    """
    try:
        # Read the image from buffer data (NumPy array)
        img_array = np.asarray(bytearray(buffer_data), dtype=np.uint8)
        img = cv2.imdecode(
            img_array, cv2.IMREAD_GRAYSCALE
        )  # Decode the image as grayscale

        return img
    except Exception as e:
        print("Error:", e)
        return None


# Save Image (Format)
def save(image, filename, format=".jpg"):
    """
    Saves the image to the specified file and format.
    :param image: Input image
    :param filename: Name of the file (without extension)
    :param format: Image format (default is .jpg)
    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot save.")
        if not filename:
            raise ValueError("Filename must be specified.")
        valid_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        if format not in valid_formats:
            raise ValueError(
                f"Invalid format '{format}'. Supported formats: {valid_formats}"
            )

        save_path = f"{filename}{format}"
        cv2.imwrite(save_path, image)
    except Exception as e:
        print(f"Error saving image: {e}")


def resize(
    image, size: Optional[Union[int, Tuple[int, int]]] = None, is_mask: bool = False
):
    """
    Resizes an image while optionally preserving the aspect ratio.

    Args:
        image (numpy.ndarray): The input image.
        size (Optional[Union[int, Tuple[int, int]]]):
            - If an integer is provided, the image is resized while maintaining the aspect ratio.
            - If a tuple `(width, height)` is provided, the image is resized to exact dimensions.
            - If None, the image remains unchanged.
        is_mask (bool, optional):
            - If True, uses nearest-neighbor interpolation (suitable for masks).
            - Otherwise, uses area interpolation (better for shrinking images smoothly).

    Returns:
        numpy.ndarray: The resized image.

    Raises:
        TypeError: If `size` is not an integer, tuple, or None.
    """

    if size is None:
        return image

    # Handle aspect ratio preservation when size is an integer
    if isinstance(size, int):
        original_height, original_width = image.shape[:2]
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:  # Landscape image
            target_width = size
            target_height = int(target_width / aspect_ratio)
        else:  # Portrait or square image
            target_height = size
            target_width = int(target_height * aspect_ratio)

    elif isinstance(size, tuple) and len(size) == 2:
        target_width, target_height = size  # Use provided width and height

    else:
        raise TypeError(
            "size must be either an integer, a tuple (width, height), or None."
        )

    # Choose interpolation method
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA

    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


# Crop Image
def crop(image, x, y, width, height):
    """
    Crop the image to the specified region.
    :param image: Input image
    :param x: Starting x coordinate
    :param y: Starting y coordinate
    :param width: Width of the crop
    :param height: Height of the crop

    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot crop.")
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(
                "Invalid crop dimensions. Coordinates and dimensions must be positive."
            )
        if y + height > image.shape[0] or x + width > image.shape[1]:
            raise ValueError("Crop dimensions exceed image bounds.")

        cropped_image = image[y : y + height, x : x + width]
        return cropped_image
    except Exception as e:
        print(f"Error cropping image: {e}")


# Rotate Image
def rotate(image, angle):
    """
    Rotate the image by the specified angle.
    :param image: Input image
    :param angle: Rotation angle in degrees
    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot rotate.")
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated_image
    except Exception as e:
        print(f"Error rotating image: {e}")


# Flip Image
def flip(image, flip_code):
    """
    Flip the image.
    :param image: Input image
    :param flip_code: Flip code (0 = vertical, 1 = horizontal, -1 = both)
    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot flip.")
        if flip_code not in [0, 1, -1]:
            raise ValueError(
                "Invalid flip code. Use 0 (vertical), 1 (horizontal), or -1 (both)."
            )

        flipped_image = cv2.flip(image, flip_code)
        return flipped_image
    except Exception as e:
        print(f"Error flipping image: {e}")


# Color Space Conversion
def convert_color_space(image, color_space):
    """
    Convert the image to the specified color space.
    :param image: Input image
    :param color_space: Target color space (e.g., 'GRAY', 'RGB', 'HSV')
    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot convert color space.")
        if color_space == "GRAY":
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif color_space == "RGB":
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space == "HSV":
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError("Unsupported color space. Use 'GRAY', 'RGB', or 'HSV'.")

        return converted_image
    except Exception as e:
        print(f"Error converting color space: {e}")


def crop_with_pad(
    image: np.ndarray, left: int, top: int, right: int, bottom: int, pad: float = 0.0
) -> np.ndarray:
    """
    Crop a region from the image while retaining the image area and applying padding strictly within the bounds of the image.

    Args:
        image (np.ndarray): The input image as a NumPy array (BGR format).
        left (int): The x-coordinate of the left edge of the crop rectangle.
        top (int): The y-coordinate of the top edge of the crop rectangle.
        right (int): The x-coordinate of the right edge of the crop rectangle.
        bottom (int): The y-coordinate of the bottom edge of the crop rectangle.
        pad (float): Padding percentage to add to the crop area (value between 0 and 1).

    Returns:
        np.ndarray: The cropped image with padding applied within the image bounds.
    """
    if not (0 <= pad <= 1):
        raise ValueError("Pad must be a value between 0 and 1.")

    img_height, img_width = image.shape[:2]

    # Calculate the dimensions of the crop area
    crop_width = right - left
    crop_height = bottom - top

    # Calculate padding in pixels
    pad_x = int(crop_width * pad)
    pad_y = int(crop_height * pad)

    # Adjust crop coordinates to include padding
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(img_width, right + pad_x)
    bottom = min(img_height, bottom + pad_y)

    # Crop the image within the adjusted coordinates
    cropped_image = image[top:bottom, left:right]

    return cropped_image


def ROI_blur(image: np.ndarray, roi: tuple, blur: int) -> np.ndarray:
    """
    Apply a blur to a specific region of interest (ROI) in the image.

    Args:
        image (np.ndarray): Input image (H, W, C).
        roi (tuple): The region of interest as (x, y, width, height).
        blur (int): Kernel size for the blur (single integer, will be adjusted to the nearest odd number).

    Returns:
        np.ndarray: Image with the specified ROI blurred.
    """
    # Ensure blur is an odd number
    if blur % 2 == 0:
        blur += 1

    # Convert blur into a tuple for the kernel size
    blur_ksize = (blur, blur)

    x, y, w, h = roi

    # Extract the ROI from the image
    roi_region = image[y : y + h, x : x + w]

    # Apply the blur to the ROI
    blurred_roi = cv2.GaussianBlur(roi_region, blur_ksize, 0)

    # Copy the blurred ROI back to the original image
    result = image.copy()
    result[y : y + h, x : x + w] = blurred_roi

    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img_path = "/mnt/03modeling/Datasets/body_type/version3.5/test_rectified/tested/seda/20230210_1701.jpg"

    img = load_gray(img_path=img_path)
    img = np.array(img)
    cv2.imwrite(
        "/home/azureuser/cloudfiles/code/Users/rohit.chandra/scripts/cv_utility/image_ops/saved_image.png",
        img,
    )
