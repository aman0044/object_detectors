import os

import numpy as np
from PIL import Image


def get_img_shape(img):
    return img.size


def get_img_size(image_path):
    return np.round(os.stat(image_path).st_size / 1048576, 4)


def brightness(img):
    img_array = np.array(img)
    return np.average(np.linalg.norm(img_array, axis=2)) / np.sqrt(3)


def sharpness(img):
    array = np.asarray(img.convert("L"), dtype=np.int32)
    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness


def contrast(img):
    array = np.asarray(img.convert("L"), dtype=np.int32)
    return array.std()


def rgb_mean_std(img):
    img_array = np.array(img)
    mean = img_array.reshape(-1, img_array.shape[-1]).mean(0).tolist()
    std = img_array.reshape(-1, img_array.shape[-1]).std(0).tolist()
    return {"mean": mean, "std": std}


def get_image_array_attributes(image_path):
    img = Image.open(image_path).convert("RGB")
    img_shape = get_img_shape(img)
    brightness_score = brightness(img)
    sharpness_score = sharpness(img)
    contrast_score = contrast(img)
    rgb_mean_std_val = rgb_mean_std(img)
    image_size = get_img_size(image_path)
    ret = {
        "image_width": img_shape[0],
        "image_height": img_shape[1],
        "brightness": brightness_score,
        "sharpness": sharpness_score,
        "contrast": contrast_score,
        "rgb_mean": rgb_mean_std_val["mean"],
        "rgb_std": rgb_mean_std_val["std"],
        "size(MBs)": image_size,
    }

    return ret
