import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from image_utils import get_image_array_attributes
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm


class BDDDataset:
    defined_classes = (
        [
            "truck",
            "train",
            "person",
            "bus",
            "car",
            "rider",
            "traffic sign",
            "bike",
            "traffic light",
            "motor",
        ],
    )

    def __init__(
        self,
        json_file_path: str,
        csv_output_path: str,
        img_base_path: str,
        required_classes: Optional[List[str]] = "all",
    ):
        """
        Initializes the BDDDataset class with paths to the JSON and CSV files.

        Args:
            json_file_path (str): Path to the input JSON file.
            csv_file_path (str): Path to the output CSV file.
            img_base_path (str): Path to the directory containing the images.
            required_classes (Optional[List[str]]): List of required classes. Defaults to "all".
        """
        self.img_base_path = Path(img_base_path)
        self.json_file_path = json_file_path
        self.csv_output_path = Path(csv_output_path)
        os.makedirs(self.csv_output_path, exist_ok=True)
        self.required_classes = self.__verify_classes(required_classes)
        logger.info(f"Required classes: {self.required_classes}")
        self.json_data = self.load_json()
        self.image_features_df = None
        self.anno_features_df = None

    def __verify_classes(self, required_classes: Optional[List[str]] = "all"):
        """
        Verifies that the required classes are valid.

        Args:
            required_classes (Optional[List[str]], optional): List of required classes. Defaults to "all".

        Raises:
            ValueError: If required_classes is not 'all' or a list of class names.

        Returns:
            List[str]: List of valid class names.
        """

        if required_classes == "all":
            return BDDDataset.defined_classes[0]
        else:
            if isinstance(required_classes, list):
                return [
                    cls_val
                    for cls_val in required_classes
                    if cls_val in BDDDataset.defined_classes[0]
                ]
            else:
                raise ValueError(
                    "required_classes should be 'all' or a list of class names."
                )

    def load_json(self):
        """
        Loads the JSON file and returns the data.

        Returns:
            dict: Parsed JSON data.
        """
        try:
            logger.info(f"Loading JSON file: {self.json_file_path}")
            with open(self.json_file_path, "r") as json_file:
                data = json.load(json_file)
            return data
        except Exception as e:
            print(f"An error occurred while loading JSON: {e}")
            return None

    @staticmethod
    def _extract_image_info(data: Dict, image_base_path: Path):
        """
        A static method to extract image-level features.
        This method is picklable and can be used with joblib's Parallel.
        args:
            data (Dict): A dictionary containing JSON data.
            image_base_path (Path): Path to the directory containing the images.
        """
        try:
            image_name = data.get("name", "N/A")
            image_weather = data["attributes"].get("weather", "N/A")
            image_scene = data["attributes"].get("scene", "N/A")
            image_timeofday = data["attributes"].get("timeofday", "N/A")

            # Assuming `get_image_array_attributes` is imported
            img_features = get_image_array_attributes(image_base_path / image_name)
            img_features["image_name"] = image_name
            img_features["image_weather"] = image_weather
            img_features["image_scene"] = image_scene
            img_features["image_timeofday"] = image_timeofday
            return img_features
        except Exception as e:
            print(f"An error occurred while extracting image features: {e}")
            return None

    def get_image_features(self):
        """
        Extracts image-level features from the JSON file and saves the results to a CSV file.

        This method uses joblib's Parallel to extract image-level features in parallel. The
        features extracted include image width, height, brightness, sharpness, contrast, RGB mean,
        RGB standard deviation, and image size in bytes.

        The method also saves the image features to a CSV file at the specified output path.

        Returns:
            None
        """

        self.image_level_features = []

        executor = Parallel(n_jobs=os.cpu_count() - 1, backend="multiprocessing")

        tasks = (
            delayed(self._extract_image_info)(data, self.img_base_path)
            for data in tqdm(
                self.json_data,
                desc="Extracting image features",
                total=len(self.json_data),
            )
        )
        data = executor(tasks)
        data = [d for d in data if d is not None]
        self.image_features_df = pd.DataFrame(data)
        print("Image features extracted successfully.")
        self.image_features_df.to_csv(
            self.csv_output_path / "image_features.csv", index=False
        )
        print(f"Image features saved to {self.csv_output_path / 'image_features.csv'}")
        print(f"Image dataframe info: {self.image_features_df.info()}")

    @staticmethod
    def _extract_anno_info(item: Dict, required_classes: List[str]):
        """
        A static method to extract object detection annotations.
        This method is picklable and can be used with joblib's Parallel.
        args:
            item (Dict): A dictionary containing JSON data.
            required_classes (List[str]): A list of required classes.
        """
        try:
            # Iterate through the JSON data
            image_name = item.get("name", "N/A")

            # Extract object detection values
            labels = item.get("labels", [])
            # print(labels)
            annos = []

            for label in labels:
                category = label.get("category", "N/A")
                # print(category, required_classes)

                if category in required_classes:
                    box2d = label.get("box2d", "N/A")
                    occluded = label["attributes"].get("occluded", "N/A")
                    truncated = label["attributes"].get("truncated", "N/A")
                    trafficLightColor = label["attributes"].get(
                        "trafficLightColor", "N/A"
                    )
                    # Extract bounding box coordinates
                    x1 = box2d.get("x1", "N/A")
                    y1 = box2d.get("y1", "N/A")
                    x2 = box2d.get("x2", "N/A")
                    y2 = box2d.get("y2", "N/A")
                    # Append the row as a dictionary
                    annos.append(
                        {
                            "image_name": image_name,
                            "category": category,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "occluded": occluded,
                            "truncated": truncated,
                            "trafficLightColor": trafficLightColor,
                        }
                    )
            return annos
        except Exception as e:
            logger.error(
                f"An error occurred while extracting object detection values: {e}"
            )
            return None

    def get_od_annotations(self):
        """
        Parses a JSON file containing object label values and image metadata,
        and writes the data into a CSV file using Pandas.
        """

        executor = Parallel(n_jobs=os.cpu_count() - 1, backend="multiprocessing")

        tasks = (
            delayed(BDDDataset._extract_anno_info)(data, self.required_classes)
            for data in tqdm(
                self.json_data,
                desc="Extracting object detection annotations",
                total=len(self.json_data),
            )
        )

        data = executor(tasks)
        # print(data)
        all_anno = []
        for anno in data:
            if anno is not None:
                all_anno.extend(anno)

        self.anno_features_df = pd.DataFrame(all_anno)
        logger.info("Object detection annotations extracted successfully.")
        self.anno_features_df.to_csv(
            self.csv_output_path / "object_detection_annotations.csv", index=False
        )
        logger.info(
            f"Object detection annotations saved to {self.csv_output_path / 'object_detection_annotations.csv'}"
        )
        logger.info(f"Object detection dataframe info: {self.anno_features_df.info()}")

    def process(
        self, image_stats_flag: bool = True, annotation_stats_flag: bool = True
    ):
        """
        Processes image and annotation statistics based on specified flags.

        This method conditionally triggers the extraction of image-level features
        and object detection annotations by invoking respective methods. The operations
        performed are controlled by the boolean flags passed as arguments.

        Args:
            image_stats_flag (bool): If True, extracts image-level features using
                the `get_image_features` method. Defaults to True.
            annotation_stats_flag (bool): If True, extracts object detection annotations
                using the `get_od_annotations` method. Defaults to True.

        Returns:
            None
        """

        if image_stats_flag:
            logger.info("Extracting image features...")
            self.get_image_features()
            logger.info("Image features extracted successfully.")
        if annotation_stats_flag:
            logger.info("Extracting object detection annotations...")
            self.get_od_annotations()
            logger.info("Object detection annotations extracted successfully.")
