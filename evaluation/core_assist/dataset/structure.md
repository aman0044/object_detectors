# Object Detection Dataset Format Guide

This guide explains the directory structure and format specifications for three different object detection dataset formats: COCO, YOLO, and Base format.

## Table of Contents
- [Common Structure](#common-structure)
- [COCO Format](#coco-format)
- [YOLO Format](#yolo-format)
- [Base Format](#base-format)
- [Detectron2 Format](#detectron2-format)
- [Mask Format](#mask-format)



## Common Structure

All formats share a common base directory structure:

```
dataset_root/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    └── [format-specific structure]
```

## COCO Format

### With Splits
```
dataset_root/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── valid/
│   │   ├── image2.jpg
│   │   └── ...
│   └── test/
│       ├── image3.jpg
│       └── ...
└── annotations/
    ├── train.json
    ├── valid.json
    └── test.json
```

### Without Splits
```
dataset_root/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    └── annotations.json
```

### COCO JSON Format
```json
{
    "images": [
        {
            "id": 1,
            "file_name": "image1.jpg",
            "width": 800,
            "height": 600
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "car"
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [x, y, width, height],
            "area": 60000,
            "iscrowd": 0,
            "score": 0.95  // only for predictions
        }
    ]
}
```

## YOLO Format

### With Splits
```
dataset_root/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── valid/
│   │   ├── image2.jpg
│   │   └── ...
│   └── test/
│       ├── image3.jpg
│       └── ...
└── annotations/
    ├── dataset.yml
    ├── train/
    │   ├── image1.txt
    │   └── ...
    ├── valid/
    │   ├── image2.txt
    │   └── ...
    └── test/
        ├── image3.txt
        └── ...
```

### Without Splits
```
dataset_root/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    ├── dataset.yml
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### YOLO Format Specifications
- `classes.txt`: One class name per line
- Label files (*.txt):
  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```
  For predictions:
  ```
  <class_id> <x_center> <y_center> <width> <height> <confidence>
  ```
  - All values are normalized (0-1)
  - One object per line
  - Filename matches corresponding image name

## Base Format

### With Splits
```
dataset_root/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── valid/
│   │   ├── image2.jpg
│   │   └── ...
│   └── test/
│       ├── image3.jpg
│       └── ...
└── annotations/
    ├── train/
    │   ├── image1.json
    │   └── ...
    ├── valid/
    │   ├── image2.json
    │   └── ...
    └── test/
        ├── image3.json
        └── ...
```

### Without Splits
```
dataset_root/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    ├── image1.json
    ├── image2.json
    └── ...
```

### Base JSON Format
```json
{
    "image_path": "path/to/image1.jpg",
    "image_name": "image1",
    "height": 600,
    "width": 800,
    "annotations": [  // for ground truth
        {
            "label": "car",
            "bbox": [x_min, y_min, height, width]
        }
    ],
    "predictions": [  // for predictions
        {
            "label": "car",
            "bbox": [x_min, y_min, height, width],
            "score": 0.95
        }
    ]
}
```

## Detectron2 Format

### With Splits
```
dataset_root/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── valid/
│   │   ├── image2.jpg
│   │   └── ...
│   └── test/
│       ├── image3.jpg
│       └── ...
└── annotations/
    ├── train.json
    ├── valid.json
    └── test.json
```

### Without Splits
```
dataset_root/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    └── annotations.json
```

### Detectron2 JSON Format
```json
[
    {
        "file_name": "path/to/image1.jpg",
        "height": 480,
        "width": 640,
        "image_id": 0,  # Mapped image_id
        "annotations": [
            {
                "bbox": [100, 150, 200, 300],
                "bbox_mode": 1,
                "category_id": 0,
                "segmentation": [[100, 150, 200, 150, 200, 300, 100, 300]],  # Polygon format
                "keypoints": [],
                "iscrowd": 0,
                "image_id": 0  # Mapped image_id
            }
        ]
    },
    {
        "file_name": "path/to/image2.jpg",
        "height": 600,
        "width": 800,
        "image_id": 1,  # Mapped image_id
        "annotations": [
            {
                "bbox": [200, 250, 300, 400],
                "bbox_mode": 1,
                "category_id": 1,
                "segmentation": [[100, 150, 200, 150, 200, 300, 100, 300]],  # Polygon format
                "keypoints": [],
                "iscrowd": 0,
                "image_id": 1  # Mapped image_id
            }
        ]
    }
]
```

## Mask Format

### With Splits
```
dataset_root/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── valid/
│   │   ├── image2.jpg
│   │   └── ...
│   └── test/
│       ├── image3.jpg
│       └── ...
└── annotations/
    ├── train/
    │   ├── image1.jpg
    │   └── ...
    ├── valid/
    │   ├── image2.jpg
    │   └── ...
    └── test/
        ├── image3.jgp
        └── ...
```

### Without Splits
```
dataset_root/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
|── annotations/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
