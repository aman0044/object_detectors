import unittest

import cv2
import numpy as np
from src.preprocessing import (add_padding, blend, blend_with_mask,
                               detect_and_extract_contours, detect_edges)


class TestImageLoader(unittest.TestCase):
    @staticmethod
    def sample_images():
        """
        Generate sample images for testing.
        """
        # Create a black background and a white square foreground
        background = np.zeros((200, 200, 3), dtype=np.uint8)
        foreground = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(foreground, (25, 25), (75, 75), (255, 255, 255), -1)

        # Create a circular mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)

        return background, foreground, mask

    def test_blend(self):
        background, foreground, _ = self.sample_images()

        # Resize foreground to match background dimensions
        foreground_resized = cv2.resize(
            foreground, (background.shape[1], background.shape[0])
        )

        blended = blend(background, foreground_resized, alpha=0.6, beta=0.4, gamma=0)
        self.assertIsNotNone(blended, "Blended image should not be None")
        self.assertEqual(
            blended.shape,
            background.shape,
            "Blended image should match the background dimensions",
        )

    def test_add_padding(self):
        _, foreground, _ = self.sample_images()
        padded = add_padding(
            foreground,
            constant_padding=True,
            value=10,
            border_type="constant",
            color=(128, 128, 128),
        )
        self.assertIsNotNone(padded, "Padded image should not be None")
        self.assertEqual(
            padded.shape, (120, 120, 3), "Padding dimensions are incorrect"
        )
        self.assertTrue(
            np.all(padded[:10, :, :] == (128, 128, 128)),
            "Padding color should match the specified color",
        )

    def test_detect_edges(self):
        _, foreground, _ = self.sample_images()
        edges = detect_edges(foreground, method="canny", threshold1=50, threshold2=150)
        self.assertIsNotNone(edges, "Edge-detected image should not be None")
        self.assertEqual(
            len(edges.shape),
            2,
            "Edge-detected image should be single-channel (grayscale)",
        )

    def test_detect_and_extract_contours(self):
        _, foreground, mask = self.sample_images()
        contours, hierarchy = detect_and_extract_contours(mask, min_area=10)
        self.assertIsNotNone(contours, "Contours should not be None")
        self.assertGreater(len(contours), 0, "Contours list should not be empty")
        self.assertTrue(
            all(cv2.contourArea(c) >= 10 for c in contours),
            "All contours should meet the minimum area requirement",
        )

    def test_blend_with_mask(self):
        background, foreground, mask = self.sample_images()

        # Resize foreground and mask to match background dimensions
        foreground_resized = cv2.resize(
            foreground, (background.shape[1], background.shape[0])
        )
        mask_resized = cv2.resize(mask, (background.shape[1], background.shape[0]))

        blended = blend_with_mask(
            background,
            foreground_resized,
            mask_resized,
            position=(0, 0),
            alpha=0.5,
            beta=0.5,
        )
        self.assertIsNotNone(blended, "Blended image should not be None")
        self.assertEqual(
            blended.shape,
            background.shape,
            "Blended image should match the background dimensions",
        )


if __name__ == "__main__":
    unittest.main()
