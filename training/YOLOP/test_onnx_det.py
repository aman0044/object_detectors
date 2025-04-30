import argparse
import glob
# import pandas as pd
import json
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from lib.core.general import non_max_suppression
from loguru import logger
from tqdm import tqdm


def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh : dh + new_unpad_h, dw : dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


def load_onnx_model(weight="yolop-640-640.onnx"):
    ort.set_default_logger_severity(4)
    onnx_path = f"./weights/{weight}"
    ort_session = ort.InferenceSession(onnx_path)
    logger.info(f"Load {onnx_path} done!")

    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()

    for ii in inputs_info:
        logger.info("Input: ", ii)
    for oo in outputs_info:
        logger.info("Output: ", oo)

    logger.info("num outputs: ", len(outputs_info))
    return ort_session


class BDDModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.ort_session = load_onnx_model(model_path)

    def preprocess(self, img_bgr):

        # height, width, _ = img_bgr.shape

        # convert to RGB
        img_rgb = img_bgr[:, :, ::-1].copy()

        # resize & normalize
        canvas, r, dw, dh, _, _ = resize_unscale(img_rgb, (640, 640))

        img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)

        img = np.expand_dims(img, 0)  # (1, 3,640,640)
        return img, r, dw, dh

    def postprocess(self, det_out, r, dw, dh):
        det_out = torch.from_numpy(det_out).float()
        boxes = non_max_suppression(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
        boxes = boxes.cpu().numpy().astype(np.float32)

        if boxes.shape[0] == 0:
            logger.info("no bounding boxes detected.")
            return None

        # scale coords to original size.
        boxes[:, 0] -= dw
        boxes[:, 1] -= dh
        boxes[:, 2] -= dw
        boxes[:, 3] -= dh
        boxes[:, :4] /= r
        return boxes

    def predict(self, img):
        img, r, dw, dh = self.preprocess(img)
        det_out = self.ort_session.run(["det_out"], input_feed={"images": img})[0]
        boxes = self.postprocess(det_out, r, dw, dh)
        return boxes


def infer_yolop(
    weight="yolop-640-640.onnx",
    img_dir="./inference/images/7dd9ef45-f197db95.jpg",
    json_output_dir="./inference/onnx_det/",
):
    os.makedirs(json_output_dir, exist_ok=True)
    # load model
    model = BDDModel(weight)
    # load image paths
    image_paths = glob.glob(str(Path(img_dir) / "*.jpg"))
    logger.info(f"Total image paths: {len(image_paths)}")

    # csv_data = []
    # json_data = []
    for img_path in tqdm(image_paths):
        img_bgr = cv2.imread(img_path)
        img_name = os.path.basename(img_path)
        h, w = img_bgr.shape[:2]
        json_info = {
            "image_path": img_path,
            "image_name": img_name,
            "height": h,
            "width": w,
            "predictions": [],
        }

        # inference: (1,n,6)
        boxes = model.predict(img_bgr)
        if boxes is None:
            logger.info(f"no bounding boxes detected in {img_path}.")
            # csv_data.append({"image_name": img_name, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "conf": 0, "label": -1})

        # logger.info(f"detect {boxes.shape[0]} bounding boxes.")
        else:
            for i in range(boxes.shape[0]):
                x1, y1, x2, y2, conf, label = boxes[i]
                x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
                # csv_data.append({"image_name": img_name, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "label": label})
                json_info["predictions"].append(
                    {
                        "image_id": 1,
                        "bbox": [x1, y1, x2, y2],
                        "label": label,
                        "score": float(conf),
                    }
                )

        json_output_path = os.path.join(
            json_output_dir, img_name.replace(".jpg", ".json")
        )
        with open(json_output_path, "w") as f:
            json.dump(json_info, f)

    logger.info("inference done.")

    # save df
    # df = pd.DataFrame(csv_data)
    # df.to_csv(csv_output_path, index=False)
    # logger.info(f"save to {csv_output_path} done.")


"""
Command

python test_onnx_det.py \
    --weight yolop-640-640.onnx \
    --img_dir ./inference/images/ \
    --json_output_dir ./inference/onnx_det/
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="yolop-640-640.onnx")
    parser.add_argument("--img_dir", type=str, default="./inference/images/")
    parser.add_argument("--json_output_dir", type=str, default="./inference/onnx_det/")
    args = parser.parse_args()

    infer_yolop(
        weight=args.weight, img_dir=args.img_dir, json_output_dir=args.json_output_dir
    )
