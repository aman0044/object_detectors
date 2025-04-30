import math
import os
import warnings
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from core_assist.dataset.visual_utils import (render_grid_mpl, render_grid_mpy,
                                              render_grid_pil)
from core_assist.plot import segment
from joblib import Parallel, delayed
from pandas.api.types import is_numeric_dtype
from pycocotools import mask as mask_utils


class SegVisualizer:
    """Creates visualizer to visualize images with segmentation masks by batch size, name and index.
    Required dataframe of the dataset as input. Can show all images with segmentation masks as a video.

    Args:
        images_dir (Union[str, os.PathLike]): Path to images in the dataset
        dataframe (pd.DataFrame): Pandas dataframe which is created by ``optical.converter``. Must contain
            ``["image_id", "segmentation", "category", "class_id"]`` columns.
        split (Optional[str], optional): Split of the dataset to be visualized.
        img_size (int, optional): Image size to resize and maintain uniformity. Defaults to 512.


    """

    def __init__(
        self,
        images_dir: Union[str, os.PathLike],
        dataframe: pd.DataFrame,
        split: Optional[str] = None,
        img_size: int = 512,
        **kwargs,
    ):
        # Check images dir and dataframe
        if ".csv" not in str(images_dir):
            assert os.path.exists(
                images_dir
            ), f"Path {images_dir} does not exist. Please check."
        # assert check_num_imgs(images_dir), f"No images found in {(images_dir)}, Please check."
        req_cols = ["image_id", "segmentation", "category", "class_id"]
        self._check_df_cols(dataframe.columns.to_list(), req_cols=req_cols)

        # Initialization
        self.images_dir = images_dir
        self.resize = (img_size, img_size)
        self.original_df = dataframe.copy()
        if (
            kwargs.get("threshold", None) is not None
            and "score" in self.original_df.columns
        ):
            threshold = kwargs.get("threshold")
            assert (
                threshold > 0 and threshold <= 1
            ), f"Threshold should be between [0.,1.], but received {threshold}"
            self.original_df = self.original_df.query("score >= @threshold")
        if split is not None:
            self.original_df = self.original_df.query("split == @split")
        if self.original_df.shape[0] < 1:
            warnings.warn(
                f"There are no images to be visualized in {split}. Please check the correct split and dataframe."
            )
        self.filtered_df = self.original_df.copy()
        self.last_sequence = 0

        # Initialize class map and color class map.
        self.class_map = pd.Series(
            self.original_df.class_id.values.astype(int),
            index=self.original_df.category,
        ).to_dict()
        self.class_map = {v: k for k, v in self.class_map.items()}
        self.class_color_map = self._get_class_color_map(self.class_map)
        self.previous_batch = []
        self.previous_args = {}

        # Initialize Resizer for maintaining image uniformity
        self.resizer = Resizer(self.resize)

    def _check_df_cols(self, cols, req_cols):
        """Check if required columns are present in dataframe."""
        for col in req_cols:
            if col not in cols:
                raise AssertionError(
                    f"Some required columns are not present in the dataframe.\
                Columns required for visualizing the segmentation are {','.join(req_cols)}."
                )
        return True

    def _get_class_color_map(self, class_map):
        """Generate color map for classes."""
        colors = {}
        for class_id in class_map:
            colors[class_id] = tuple(np.random.randint(0, 255, 3).tolist())
        return colors

    def _decode_rle(self, rle, height, width):
        """Decode RLE encoding to binary mask."""
        if isinstance(rle, dict):
            return mask_utils.decode(rle)
        elif isinstance(rle, str):
            # Check if the RLE is in COCO format (starts with a number)
            if rle[0].isdigit():
                # Convert COCO RLE format to mask
                counts = [int(x) for x in rle.split()]
                rle_dict = {"counts": counts, "size": [height, width]}
                return mask_utils.decode(rle_dict)
            else:
                # Convert from other RLE format
                binary_str = "".join(["1" if i == "T" else "0" for i in rle])
                binary_vals = [int(i) for i in binary_str]
                return np.array(binary_vals).reshape(height, width)
        else:
            return np.zeros((height, width), dtype=np.uint8)

    def __getitem__(self, image_id: str, use_original: bool = False) -> Dict:
        """Fetches images and segmentation masks from the input dataframe.

        Args:
            image_id (str): Image Id to be fetched.
            use_original (bool, optional): Whether to search image in original or filtered dataframe.
                Defaults to False.

        Returns:
            Dict: Python dictionary containing queried image and segmentation information.
        """
        if use_original:
            img_df = self.original_df[self.original_df["image_id"] == image_id]
        else:
            img_df = self.filtered_df[self.filtered_df["image_id"] == image_id]

        image_path = None
        masks = []
        mask_labels = []
        scores = []

        for row in img_df.to_dict("records"):
            is_valid_row = "segmentation" in row and row["segmentation"] is not None

            if not image_path and "image_path" in row:
                image_path = row["image_path"]

            if is_valid_row:
                # Store class label
                mask_labels.append(self.class_map[row["class_id"]])

                # Handle RLE encoded segmentation
                height = row.get("height", self.resize[0])
                width = row.get("width", self.resize[1])
                mask = self._decode_rle(row["segmentation"], height, width)
                masks.append(mask)

                # Add score if available
                if "score" in row:
                    scores.append(row["score"])

        # Ensure we have an image path
        if not image_path:
            image_path = os.path.join(self.images_dir, image_id)

        # Load and resize image
        try:
            img = self.resizer.load_image(image_path)
            # Also resize masks if needed
            resized_masks = []
            for mask in masks:
                resized_mask = self.resizer.resize_mask(mask)
                resized_masks.append(resized_mask)

            item = {
                "img": {"image_name": image_path, "image": img},
                "masks": resized_masks,
                "mask_labels": mask_labels,
            }

            if scores:
                item["scores"] = scores

            return item
        except Exception as e:
            warnings.warn(f"Error loading image {image_path}: {str(e)}")
            return None

    def _get_batch(
        self,
        samples: int = 1,
        index: Optional[int] = None,
        name: Optional[str] = None,
        random: bool = True,
        do_filter: bool = False,
        **kwargs,
    ) -> List[Dict]:
        """Fetches batch of images and segmentation masks, applies filters if provided.

        Args:
            samples (int, optional): Number of images and masks to be fetched.
                if it is ``-1`` all images and masks in the dataset will be returned.
                Defaults to 1.
            index (Optional[int], optional): Index of the image to be fetched. Defaults to None.
            name (Optional[str], optional): Name of the image to be fetched. Defaults to None.
            random (bool, optional): If ``True`` randomly selects ``samples`` images otherwise follows a sequence.
                Defaults to True.
            do_filter (bool, optional): Whether to apply filtering or not. Defaults to False.

        Returns:
            List[Dict]: List of images and segmentation info.
        """
        self.filtered_df = (
            self._apply_filters(**kwargs) if do_filter else self.filtered_df
        )
        unique_images = list(self.filtered_df.image_id.unique())
        use_original = kwargs.get("use_original", False)
        batch_img_indices = []

        if samples == -1:
            batch_img_indices = (
                list(self.original_df.image_id.unique())
                if use_original
                else unique_images
            )

        elif index is not None or name is not None:
            unique_images_original = list(self.original_df.image_id.unique())
            if index is not None:
                index = index % len(unique_images_original)
                batch_img_indices = [unique_images_original[index]]
                use_original = True
            elif name is not None and name in unique_images_original:
                batch_img_indices = [name]
                use_original = True
            else:
                print(f"{name} not found in the dataset. Please check")

        else:
            actual_num_images = min(len(unique_images), samples)
            if actual_num_images < samples:
                warnings.warn(f"Found only {actual_num_images} in the dataset.")
            if random:
                shuffle(unique_images)
                batch_img_indices = unique_images[:actual_num_images]
            else:
                start_index = self.last_sequence
                end_index = self.last_sequence + actual_num_images
                self.last_sequence = end_index if end_index <= len(unique_images) else 0
                batch_img_indices = unique_images[start_index:end_index]

        backend = "threading"
        r = Parallel(n_jobs=-1, backend=backend)(
            delayed(self.__getitem__)(idx, use_original) for idx in batch_img_indices
        )
        # Filter out None values (failed image loads)
        return [item for item in r if item is not None]

    def show_batch(
        self,
        samples: int = 9,
        previous: bool = False,
        save_path: Optional[str] = None,
        render: str = "pil",
        random: bool = True,
        **kwargs,
    ) -> Any:
        """Displays a batch of images based on input size.

        Args:
            samples (int, optional): Number of images and their masks to be visualized. Defaults to 9.
            previous (bool, optional): If ``True`` just displays last batch. Defaults to False.
            save_path (Optional[str], optional): Output path if images and masks to be saved. Defaults to None.
            render (str, optional): Rendering to be used. Available options are ``mpl``,``pil``,``mpy``.
                If ``mpl``, uses ``matplotlib`` to display the images and masks.
                If ``pil``, uses ``Pillow`` to display the images and masks.
                If ``mpy``, uses ``mediapy`` to display as video
                Defaults to "pil".
            random (bool, optional): If ``True`` randomly selects ``samples`` images otherwise follows a sequence.
                Defaults to True.

        Returns:
            Any: Incase of Pillow or mediapy rendering IPython media object will be returned.
        """
        if previous and len(self.previous_batch):
            batch = self.previous_batch
        else:
            do_filter = True
            if kwargs == self.previous_args:
                do_filter = False
            batch = self._get_batch(
                samples, random=random, do_filter=do_filter, **kwargs
            )
            self.previous_batch = batch
            self.previous_args = kwargs

        drawn_imgs, image_names = self._draw_images(batch, **kwargs)
        if samples != len(drawn_imgs):
            samples = len(drawn_imgs)
            warnings.warn(f"Visualizing only {samples} images.")

        if samples == 1:
            if save_path is not None:
                save_path = self._check_save_path(save_path)
                drawn_imgs[0].save(image_names[0] + "_vis.jpg")
            return drawn_imgs[0]

        if len(drawn_imgs) > 0:
            return self._render_image_grid(
                samples, drawn_imgs, image_names, render, save_path=save_path, **kwargs
            )
        else:
            warnings.warn("No valid images found to visualize.")
            return

    def _check_save_path(self, save_path, image_name=None):
        """Check if save path exists, if not create it."""
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if image_name:
            base_name = os.path.basename(image_name)
            save_path = os.path.join(save_path, f"{base_name}_vis.jpg")
        return save_path

    def reset_filters(self):
        """Resets all the filters applied on original dataframe."""
        self.filtered_df = self.original_df.copy()

    def _render_image_grid(
        self,
        samples: int,
        drawn_imgs: List,
        image_names: List[str],
        render: str = "pil",
        save_path: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Renders image and mask grid based on given backend.

        Args:
            samples (int): Number of images in the grid.
            drawn_imgs (List): List of images with masks.
            image_names (List[str]): List of image names in the grid.
            render (str, optional): Rendering to be used. Available options are ``mpl``,``pil``,``mpy``.
                If ``mpl``, uses ``matplotlib`` to display the images and masks.
                If ``pil``, uses ``Pillow`` to display the images and masks.
                If ``mpy``, uses ``mediapy`` to display as video
                Defaults to "pil".
            save_path (Optional[str], optional): Output path if images and masks to be saved. Defaults to None.

        Raises:
            RuntimeError: Raised if invalid rendering backend is given.

        Returns:
            Any: Incase of Pillow or mediapy rendering IPython media object will be returned.
        """
        cols = 2 if samples <= 6 else 3
        cols = 1 if samples == 1 else cols
        rows = math.ceil(samples / cols)
        if render.lower() == "mpl":
            return render_grid_mpl(
                drawn_imgs,
                image_names,
                samples,
                cols,
                rows,
                self.resize[0],
                save_path,
                **kwargs,
            )
        elif render.lower() == "pil":
            return render_grid_pil(
                drawn_imgs,
                image_names,
                samples,
                cols,
                rows,
                self.resize[0],
                save_path,
                **kwargs,
            )
        elif render.lower() == "mpy":
            return render_grid_mpy(drawn_imgs, image_names, **kwargs)
        else:
            raise RuntimeError(
                "Invalid Image grid rendering format, should be either mpl or pil."
            )

    def _draw_images(self, batch: List[Dict], **kwargs) -> Tuple[List, List]:
        """Draws segmentation masks on the images.

        Args:
            batch (List[Dict]): List of images and segmentation info.

        Returns:
            Tuple[List, List]: Tuple of drawn images and their respective image names.
        """
        drawn_imgs = []
        image_names = []

        # Get kwargs for segment function
        segment_kwargs = {
            "bbox_flag": kwargs.get("bbox_flag", False),
            "pad_bbox": kwargs.get("pad_bbox", 0.0),
            "text_color": kwargs.get("text_color", (255, 255, 255)),
            "thickness": kwargs.get("thickness", 2),
            "font_scale": kwargs.get("font_scale", 1),
            "segment_type": kwargs.get("segment_type", "both"),
            "ret": True,
        }

        for img_info in batch:
            img_name = img_info["img"]["image_name"]
            img = img_info["img"]["image"]
            masks = img_info["masks"]
            mask_labels = img_info["mask_labels"]
            scores = img_info.get("scores", None)

            try:
                # Use the provided segment function
                drawn_img, _ = segment(
                    img=np.array(img),
                    masks=masks,
                    mask_labels=mask_labels,
                    confs=scores,
                    **segment_kwargs,
                )

                image_names.append(img_name)
                drawn_imgs.append(drawn_img)
            except Exception as e:
                warnings.warn(
                    f"Could not draw segmentation masks for {img_name}: {str(e)}"
                )
                continue

        return drawn_imgs, image_names

    def _apply_filters(self, **kwargs) -> pd.DataFrame:
        """Applies filters on the original dataframe.

        Keyword Args:
            only_without_labels(bool): To filter images which do not have any segmentation masks.
            only_with_labels(bool): To filter only images which have segmentation masks.
            filter_categories(Union[str,List[]]): To filter masks with given categories.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        if kwargs.get("only_without_labels", None):
            df = self.original_df[
                self.original_df["class_id"].isna()
                & self.original_df["category"].isna()
            ]
            return df

        curr_df = self.original_df.copy()

        if kwargs.get("only_with_labels", None):
            curr_df = self.original_df.dropna(
                subset=["segmentation", "class_id", "category"], how="any"
            )

        if kwargs.get("filter_categories", None):
            filter_labels = kwargs["filter_categories"]
            ds_classes = [
                cat.lower() for cat in list(self.original_df.category.unique())
            ]
            labels = []

            if len(filter_labels) > 0:
                labels = (
                    [filter_labels] if isinstance(filter_labels, str) else filter_labels
                )
                labels = [cat.lower() for cat in labels]
                for label in labels:
                    if label not in ds_classes:
                        warnings.warn(
                            f"{label} category is not present in the dataset. Please check"
                        )

            if len(labels) > 0:
                curr_df = curr_df[curr_df["category"].str.lower().isin(labels)]

        return curr_df

    def show_image(
        self,
        index: int = 0,
        name: Optional[str] = None,
        save_path: Optional[str] = None,
        render: str = "mpl",
        **kwargs,
    ) -> Any:
        """Displays images with segmentation mask given index or name.

        Args:
            index (int, optional): Index of the image to be fetched. Defaults to 0.
            name (Optional[str], optional): Name of the image to be fetched. Defaults to None.
            save_path (Optional[str], optional): Output path if images and masks to be saved. Defaults to None.
            render (str, optional): Rendering to be used. Available options are ``mpl``,``pil``,``mpy``.
                If ``mpl``, uses ``matplotlib`` to display the images and masks.
                If ``pil``, uses ``Pillow`` to display the images and masks.
                If ``mpy``, uses ``mediapy`` to display as video
                Defaults to "pil".

        Returns:
            Any: Incase of Pillow or mediapy rendering IPython media object will be returned.
        """
        if name is not None:
            batch = self._get_batch(index=None, name=name, **kwargs)
        else:
            batch = self._get_batch(index=index, name=None, **kwargs)

        drawn_img, image_name = self._draw_images(batch, **kwargs)
        if save_path is not None and len(drawn_img) > 0:
            save_path = self._check_save_path(save_path, image_name[0])
            drawn_img[0].save(save_path)

        if len(drawn_img) > 0:
            return self._render_image_grid(1, drawn_img, image_name, render, **kwargs)
        else:
            warnings.warn("No valid images found to visualize.")
            return

    def show_video(self, samples=10, use_original: bool = True, **kwargs) -> Any:
        """Displays whole dataset as a video.

        Args:
            use_original(bool): Whether to use original dataset or filtered dataset. Defaults to ``True``

        Keyword Args:
            show_image_name(bool): Whether to show image names in the video or not.
            image_time(float): How many seconds each should be displayed in the video.
                e.g: ``image_time = 1`` means each image will be displayed for one second.
                ``image_time = 0.5`` means each image will be displayed for half a second.

        Returns:
            Any: Returns IPython media object.
        """
        batch = self._get_batch(samples=samples, use_original=use_original, **kwargs)
        drawn_imgs, image_names = self._draw_images(batch, **kwargs)

        if len(drawn_imgs) > 0:
            return self._render_image_grid(
                len(drawn_imgs), drawn_imgs, image_names, render="mpy", **kwargs
            )
        else:
            warnings.warn("No valid images found to visualize.")
            return


class Resizer:
    """Utility class to resize images and masks to a fixed size."""

    def __init__(self, size):
        """Initialize with target size."""
        self.size = size

    def __call__(self, item):
        """Resize image and annotations in the item."""
        image_path = item.get("image_path")
        image = self.load_image(image_path)

        if "anns" in item:
            anns = item["anns"]
            # Scale annotations here if needed
            return image, anns

        return image

    def load_image(self, image_path):
        """Load and resize an image."""
        import cv2
        from PIL import Image

        try:
            # Load image
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image
            img = cv2.resize(img, self.size)

            # Convert to PIL Image for compatibility
            return Image.fromarray(img)
        except Exception as e:
            warnings.warn(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image in case of error
            return Image.new("RGB", self.size, (0, 0, 0))

    def resize_mask(self, mask):
        """Resize a mask."""
        import cv2

        # Ensure mask is binary
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Resize mask using nearest neighbor to preserve binary values
        resized_mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        return resized_mask
