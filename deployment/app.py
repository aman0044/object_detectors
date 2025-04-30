import io

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from model import BDDModel
from PIL import Image

app = FastAPI(title="Object Detection API")

# Load model once at startup (adjust path if using custom weights)
weight = "../training/YOLOP/weights/yolop-640-640.onnx"
model = BDDModel(weight)
class_map = {0: "vehicle"}


@app.get("/")
def root():
    return {"message": "Welcome to the YOLOv8 Object Detection API"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run inference
        results = model.predict(image)
        detections = []
        if results is None:
            logger.info(f"no bounding boxes detected")
            # csv_data.append({"image_name": img_name, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "conf": 0, "label": -1})

        # logger.info(f"detect {boxes.shape[0]} bounding boxes.")
        else:
            for i in range(results.shape[0]):
                x1, y1, x2, y2, confidence, label = results[i]
                x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
                class_name = class_map[label]
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return {"filename": file.filename, "detections": detections}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
