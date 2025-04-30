import json
import os


def convert_bdd_to_yolo(bdd_json_path, output_dir, class_map, image_size=(1280, 720)):
    """
    Convert BDD100K object detection JSON annotations to YOLO format.

    Parameters:
        bdd_json_path (str): Path to the BDD JSON annotation file.
        output_dir (str): Directory to save YOLO .txt annotation files.
        class_map (dict): Mapping from BDD class names to YOLO class IDs.
        image_size (tuple): (width, height) of the images, default is (1280, 720) for BDD100K.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(bdd_json_path, "r") as f:
        data = json.load(f)

    for item in data:
        image_name = os.path.splitext(item["name"])[0]
        txt_file = os.path.join(output_dir, image_name + ".txt")
        lines = []
        for label in item.get("labels", []):
            category = label.get("category")
            if category not in class_map:
                continue
            class_id = class_map[category]
            box2d = label.get("box2d")
            if not box2d:
                continue
            x1, y1 = box2d["x1"], box2d["y1"]
            x2, y2 = box2d["x2"], box2d["y2"]
            x_center = ((x1 + x2) / 2) / image_size[0]
            y_center = ((y1 + y2) / 2) / image_size[1]
            width = (x2 - x1) / image_size[0]
            height = (y2 - y1) / image_size[1]
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            lines.append(line)

        with open(txt_file, "w") as f:
            f.write("\n".join(lines))


# Command to run the script
# Save this file as `bdd_to_yolo.py` and use the following command in the terminal
"""
python bdd_to_yolo.py \
    --bdd_json_path ./data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
    --output_dir ./data/yolo 
    --class_map ./data_processing/data_creation/yolo_class_map.txt
"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bdd_json_path", type=str, help="Path to the BDD JSON annotation file."
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save YOLO .txt annotation files."
    )
    parser.add_argument("--class_map", type=str, help="Path to the class mapping file.")
    args = parser.parse_args()

    bdd_json_path = args.bdd_json_path
    output_dir = args.output_dir
    class_map_path = args.class_map

    class_map = {}
    with open(class_map_path, "r") as f:
        for line in f:
            class_name, class_id = line.strip().split()
            class_map[class_name] = int(class_id)
    convert_bdd_to_yolo(bdd_json_path, output_dir, class_map, image_size=(1280, 720))
