import cv2
import numpy as np
import onnxruntime as ort
import torch
from general import non_max_suppression
from loguru import logger


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
