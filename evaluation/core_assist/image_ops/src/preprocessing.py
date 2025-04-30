import cv2
import numpy as np


def blend(image1, image2, alpha, beta, gamma=0):
    """
    Blend two images together using specified weights.
    :param image1: First input image.
    :param image2: Second input image.
    :param alpha: Weight for the first image.
    :param beta: Weight for the second image.
    :param gamma: Scalar added to each sum (default is 0).
    :return: Blended image.
    """
    try:
        if image1 is None or image2 is None:
            raise ValueError("One or both input images are None. Cannot blend.")

        if image1.shape != image2.shape:
            raise ValueError("Input images must have the same dimensions for blending.")

        if not (0 <= alpha <= 1) or not (0 <= beta <= 1):
            raise ValueError("Alpha and beta must be between 0 and 1.")

        # Perform blending
        blended_image = cv2.addWeighted(image1, alpha, image2, beta, gamma)
        return blended_image

    except Exception as e:
        print(f"Error blending images: {e}")
        return None


def add_padding(
    image,
    constant_value=None,
    top=0,
    bottom=0,
    left=0,
    right=0,
    border_type="constant",
    color=(0, 0, 0),
):
    """
    Add padding to an image with the specified border type.

    :param image: Input image (numpy.ndarray).
    :param constant_padding: If True, adds `value` pixels to all sides.
    :param value: Number of pixels to add to all sides if constant_padding is True.
    :param top: Pixels to add to the top border (ignored if constant_padding is True).
    :param bottom: Pixels to add to the bottom border (ignored if constant_padding is True).
    :param left: Pixels to add to the left border (ignored if constant_padding is True).
    :param right: Pixels to add to the right border (ignored if constant_padding is True).
    :param border_type: Type of border ('constant', 'reflect', 'replicate', 'wrap').
    :param color: Color of the padding for 'constant' border type (default is black).
    :return: Padded image (numpy.ndarray) or None if an error occurs.
    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot add padding.")

        if constant_value:
            top = bottom = left = right = constant_value

        # Map border type to OpenCV constants
        border_types = {
            "constant": cv2.BORDER_CONSTANT,
            "reflect": cv2.BORDER_REFLECT,
            "replicate": cv2.BORDER_REPLICATE,
            "wrap": cv2.BORDER_WRAP,
        }

        if border_type not in border_types:
            raise ValueError(
                f"Unsupported border type '{border_type}'. Supported types: {list(border_types.keys())}"
            )

        # Add padding
        padded_image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=border_types[border_type],
            value=color if border_type == "constant" else None,
        )
        return padded_image

    except Exception as e:
        print(f"Error adding padding: {e}")
        return None


def detect_edges(image, method="canny", threshold1=100, threshold2=200, ksize=3):
    """
    Detect edges in an image using various methods.
    :param image: Input image (should be in grayscale for most methods).
    :param method: Edge detection method ('canny', 'sobel', 'laplacian').
    :param threshold1: First threshold for edge detection (used in 'canny').
    :param threshold2: Second threshold for edge detection (used in 'canny').
    :param ksize: Kernel size for Sobel/Laplacian filters (must be odd and greater than 1).
    :return: Image with detected edges.
    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot perform edge detection.")

        if method not in ["canny", "sobel", "laplacian"]:
            raise ValueError(
                f"Unsupported method '{method}'. Supported methods: 'canny', 'sobel', 'laplacian'."
            )

        # Ensure the input image is grayscale
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Apply the chosen edge detection method
        if method == "canny":
            edges = cv2.Canny(gray_image, threshold1, threshold2)
        elif method == "sobel":
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)  # Sobel X
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)  # Sobel Y
            edges = cv2.magnitude(sobelx, sobely).astype(np.uint8)  # Combine gradients
        elif method == "laplacian":
            edges = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=ksize).astype(np.uint8)

        return edges

    except Exception as e:
        print(f"Error detecting edges: {e}")
        return None


def extract_contours(
    image, min_area=100, retrieval_mode="external", approximation_mode="simple"
):
    """
    Detect and extract contours from an image.
    :param image: Input binary or grayscale image.
    :param min_area: Minimum area of the contour to be considered.
    :param retrieval_mode: Contour retrieval mode ('external' or 'tree').
    :param approximation_mode: Contour approximation ('simple' or 'none').
    :return: List of contours and hierarchy.
    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot detect contours.")

        # Ensure input image is grayscale
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Map retrieval and approximation modes to OpenCV constants
        retrieval_modes = {"external": cv2.RETR_EXTERNAL, "tree": cv2.RETR_TREE}

        approximation_modes = {
            "simple": cv2.CHAIN_APPROX_SIMPLE,
            "none": cv2.CHAIN_APPROX_NONE,
        }

        if (
            retrieval_mode not in retrieval_modes
            or approximation_mode not in approximation_modes
        ):
            raise ValueError("Invalid retrieval or approximation mode specified.")

        # Detect contours
        contours, hierarchy = cv2.findContours(
            gray_image,
            retrieval_modes[retrieval_mode],
            approximation_modes[approximation_mode],
        )

        # Filter contours by minimum area
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        return contours, hierarchy

    except Exception as e:
        print(f"Error detecting contours: {e}")
        return None, None


def blend_with_mask(
    background, foreground, mask, position=(0, 0), alpha=0.5, beta=0.5, gamma=0
):
    """
    Blend a foreground image onto a background using a mask.

    :param background: Background image (numpy array).
    :param foreground: Foreground image (numpy array).
    :param mask: Binary mask for blending (numpy array).
    :param position: Top-left corner (x, y) to place the foreground on the background.
    :param alpha: Weight of the foreground in blending (0 to 1).
    :param beta: Weight of the background in blending (0 to 1).
    :param gamma: Scalar added to the blended result.
    :return: Blended image (numpy array).
    """
    try:
        if background is None or foreground is None or mask is None:
            raise ValueError("Background, foreground, or mask is None. Cannot blend.")

        if not (0 <= alpha <= 1) or not (0 <= beta <= 1):
            raise ValueError("Alpha and beta must be between 0 and 1.")

        # Get dimensions
        x, y = position
        fh, fw = foreground.shape[:2]  # Foreground height and width
        mh, mw = mask.shape[:2]  # Mask height and width

        if fw != mw or fh != mh:
            raise ValueError("Foreground and mask dimensions do not match.")

        # Ensure the region fits within the background dimensions
        if (
            x < 0
            or y < 0
            or x + fw > background.shape[1]
            or y + fh > background.shape[0]
        ):
            raise ValueError(
                "Foreground image and mask exceed background dimensions at the given position."
            )

        # Convert mask to 3 channels if necessary
        if len(mask.shape) == 2:  # Grayscale mask
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Extract ROI from the background
        roi = background[y : y + fh, x : x + fw]

        # Blend using the mask
        blended_foreground = cv2.bitwise_and(foreground, mask)
        blended_background = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
        blended_image = cv2.addWeighted(
            src1=blended_foreground,
            alpha=alpha,
            src2=blended_background,
            beta=beta,
            gamma=gamma,
        )

        # Place the blended result back onto the background
        result = background.copy()
        result[y : y + fh, x : x + fw] = blended_image

        return result

    except Exception as e:
        print(f"Error blending with mask: {e}")
        return None
