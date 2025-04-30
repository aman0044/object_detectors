import os
import unittest

import cv2
import numpy as np
from image_ops.src.image_utils import (convert_color_space, crop, flip,
                                       load_bgr, load_buffer_bgr,
                                       load_buffer_gray, load_buffer_rgb,
                                       load_gray, load_img, load_rgb, resize,
                                       rotate, save)


class TestImageOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up resources for testing.
        """
        cls.test_image_path = "/home/azureuser/cloudfiles/code/Users/rohit.chandra/scripts/cv_utility/examples/test_image/gettyimages-1164166864-612x612.jpg"
        cls.sample_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Black test image
        cls.buffer_image = cv2.imencode(".jpg", cls.sample_image)[1].tobytes()
        cv2.imwrite(cls.test_image_path, cls.sample_image)  # Save to disk for testing

    def test_load_img(self):
        """
        Test loading an image from a file.
        """
        img = load_img(self.test_image_path)
        self.assertIsNotNone(img, "Image failed to load.")
        self.assertEqual(
            img.shape, self.sample_image.shape, "Loaded image dimensions are incorrect."
        )

    def test_load_rgb(self):
        """
        Test loading an image in RGB format.
        """
        img = load_rgb(self.test_image_path)
        self.assertIsNotNone(img, "RGB image failed to load.")
        self.assertEqual(
            img.shape, self.sample_image.shape, "RGB image dimensions are incorrect."
        )

    def test_load_bgr(self):
        """
        Test loading an image in BGR format.
        """
        img = load_bgr(self.test_image_path)
        self.assertIsNotNone(img, "BGR image failed to load.")
        self.assertEqual(
            img.shape, self.sample_image.shape, "BGR image dimensions are incorrect."
        )

    def test_load_gray(self):
        """
        Test loading an image in grayscale format.
        """
        img = load_gray(self.test_image_path)
        self.assertIsNotNone(img, "Gray image failed to load.")
        self.assertEqual(len(img.shape), 2, "Grayscale image should have 2 dimensions.")

    def test_load_buffer_rgb(self):
        """
        Test loading an RGB image from a buffer.
        """
        img = load_buffer_rgb(self.buffer_image)
        self.assertIsNotNone(img, "Failed to load RGB image from buffer.")
        self.assertEqual(
            img.shape,
            self.sample_image.shape,
            "RGB image dimensions from buffer are incorrect.",
        )

    def test_load_buffer_bgr(self):
        """
        Test loading a BGR image from a buffer.
        """
        img = load_buffer_bgr(self.buffer_image)
        self.assertIsNotNone(img, "Failed to load BGR image from buffer.")
        self.assertEqual(
            img.shape,
            self.sample_image.shape,
            "BGR image dimensions from buffer are incorrect.",
        )

    def test_load_buffer_gray(self):
        """
        Test loading a grayscale image from a buffer.
        """
        img = load_buffer_gray(self.buffer_image)
        self.assertIsNotNone(img, "Failed to load grayscale image from buffer.")
        self.assertEqual(
            len(img.shape), 2, "Grayscale image from buffer should have 2 dimensions."
        )

    def test_save(self):
        """
        Test saving an image to a file.
        """
        save(self.sample_image, "test_save", format=".png")
        self.assertTrue(os.path.exists("test_save.png"), "Image file was not saved.")
        os.remove("test_save.png")  # Cleanup

    def test_resize(self):
        """
        Test resizing an image.
        """
        resized_img = resize(self.sample_image, width=50, height=50)
        self.assertEqual(
            resized_img.shape, (50, 50, 3), "Resized image dimensions are incorrect."
        )

    def test_crop(self):
        """
        Test cropping an image.
        """
        cropped_img = crop(self.sample_image, x=10, y=10, width=50, height=50)
        self.assertEqual(
            cropped_img.shape, (50, 50, 3), "Cropped image dimensions are incorrect."
        )

    def test_rotate(self):
        """
        Test rotating an image.
        """
        rotated_img = rotate(self.sample_image, angle=90)
        self.assertEqual(
            rotated_img.shape,
            self.sample_image.shape,
            "Rotated image dimensions are incorrect.",
        )

    def test_flip(self):
        """
        Test flipping an image.
        """
        flipped_img = flip(self.sample_image, flip_code=1)
        self.assertEqual(
            flipped_img.shape,
            self.sample_image.shape,
            "Flipped image dimensions are incorrect.",
        )

    def test_convert_color_space(self):
        """
        Test converting an image to different color spaces.
        """
        gray_img = convert_color_space(self.sample_image, color_space="GRAY")
        self.assertEqual(len(gray_img.shape), 2, "Grayscale conversion failed.")
        hsv_img = convert_color_space(self.sample_image, color_space="HSV")
        self.assertEqual(
            hsv_img.shape, self.sample_image.shape, "HSV conversion failed."
        )

    # def test_invalid_inputs(self):
    #     """
    #     Test invalid inputs for various functions.
    #     """
    #     # Test invalid image inputs
    #     with self.assertRaises(ValueError):
    #         save(None, "test_invalid")
    #     with self.assertRaises(ValueError):
    #         resize(None, width=50, height=50)
    #     with self.assertRaises(ValueError):
    #         crop(None, x=10, y=10, width=50, height=50)
    #     with self.assertRaises(ValueError):
    #         rotate(None, angle=90)
    #     with self.assertRaises(ValueError):
    #         flip(None, flip_code=1)
    #     with self.assertRaises(ValueError):
    #         convert_color_space(None, color_space="GRAY")

    # # Test invalid parameters
    # with self.assertRaises(ValueError):
    #     resize(self.sample_image, width=None, height=None)
    # with self.assertRaises(ValueError):
    #     crop(self.sample_image, x=90, y=90, width=20, height=20)  # Out-of-bounds crop
    # with self.assertRaises(ValueError):
    #     flip(self.sample_image, flip_code=2)  # Invalid flip code
    # with self.assertRaises(ValueError):
    #     convert_color_space(self.sample_image, color_space="XYZ")  # Unsupported color space

    @classmethod
    def tearDownClass(cls):
        """
        Clean up resources after testing.
        """
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)


if __name__ == "__main__":
    unittest.main()
