# Matrix Module Structure Documentation

This document describes the structure of the JSON files used for classification, detection, and segmentation tasks in the matrix module.

---

## 1. Classification

### Prediction JSON (`pred_json`)
The prediction JSON for classification contains the following fields:

```json
{
    "image_path": "string",          // Path to the image file
    "image_name": "string",          // Name of the image file
    "predictions": {
        "label": ["string"],         // Predicted label(s)
        "score": [float]             // Confidence score(s) for the predicted label(s)
    }
}
```

### Ground Truth JSON (`gt_json`)
The ground truth JSON for classification contains the following fields:

```json
{
    "image_path": "string",          // Path to the image file
    "image_name": "string",          // Name of the image file
    "annotations": {
        "label": ["string"]          // Ground truth label(s)
    }
}
```

---

## 2. Detection

### Prediction JSON (`pred_json`)
The prediction JSON for detection contains the following fields:

```json
{
    "image_path": "string",          // Path to the image file
    "image_name": "string",          // Name of the image file
    "height": int,                   // Height of the image
    "width": int,                    // Width of the image
    "predictions": [
        {
            "bbox": [x1, y1, x2, y2], // Bounding box coordinates (x1, y1, x2, y2)
            "label": "string",       // Predicted label for the bounding box
            "score": float           // Confidence score for the prediction
        }
    ]
}
```

### Ground Truth JSON (`gt_json`)
The ground truth JSON for detection contains the following fields:

```json
{
    "image_path": "string",          // Path to the image file
    "image_name": "string",          // Name of the image file
    "height": int,                   // Height of the image
    "width": int,                    // Width of the image
    "annotations": [
        {
            "label": "string",       // Ground truth label for the bounding box
            "bbox": [x1, y1, x2, y2] // Bounding box coordinates (x1, y1, x2, y2)
        }
    ]
}
```

---

## 3. Segmentation

### Prediction JSON (`pred_json`)
The prediction JSON for segmentation contains the following fields:

```json
{
    "image_path": "string",          // Path to the image file
    "image_name": "string",          // Name of the image file
    "height": int,                   // Height of the image
    "width": int,                    // Width of the image
    "predictions": [
        {
            "image_id": "string",    // Unique identifier for the image
            "category_id": int,      // ID of the predicted category
            "category_name": "string", // Name of the predicted category
            "segmentation": {
                "size": [height, width], // Size of the segmentation mask
                "counts": "string"       // RLE-encoded segmentation mask
            },
            "bbox": [x1, y1, x2, y2], // Bounding box coordinates (x1, y1, x2, y2)
            "score": float           // Confidence score for the prediction
        }
    ]
}
```

### Ground Truth JSON (`gt_json`)
The ground truth JSON for segmentation contains the following fields:

```json
{
    "image_path": "string",          // Path to the image file
    "image_name": "string",          // Name of the image file
    "height": int,                   // Height of the image
    "width": int,                    // Width of the image
    "annotations": [
        {
            "label": "string",       // Ground truth label for the segmentation
            "segmentation": {
                "size": [height, width], // Size of the segmentation mask
                "counts": "string"       // RLE-encoded segmentation mask
            }
        }
    ]
}
```

