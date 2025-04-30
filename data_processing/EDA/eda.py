import argparse

from dataset import BDDDataset

"""
Command to run this file:
python eda.py --json_file_path ./data/bdd100k_labels_images_val.json \
    --csv_output_path ./data/val \
    --img_base_path ./data/bdd100k_images_100k \
    --required_classes "all" \
    --image_stats_flag \
    --annotation_stats_flag 
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BDD dataset for EDA")
    parser.add_argument(
        "--json_file_path", type=str, required=True, help="Path to the input JSON file"
    )
    parser.add_argument(
        "--csv_output_path",
        type=str,
        required=True,
        help="Path to the output CSV directory",
    )
    parser.add_argument(
        "--img_base_path",
        type=str,
        required=True,
        help="Path to the directory containing the images",
    )
    parser.add_argument(
        "--required_classes",
        type=str,
        nargs="+",
        default="all",
        help="List of required classes",
    )
    parser.add_argument(
        "--image_stats_flag",
        action="store_true",
        help="Flag to enable image statistics",
    )
    parser.add_argument(
        "--annotation_stats_flag",
        action="store_true",
        help="Flag to enable annotation statistics",
    )
    args = parser.parse_args()

    dataset = BDDDataset(
        json_file_path=args.json_file_path,
        csv_output_path=args.csv_output_path,
        img_base_path=args.img_base_path,
        required_classes=args.required_classes,
    )
    dataset.process(
        image_stats_flag=args.image_stats_flag,
        annotation_stats_flag=args.annotation_stats_flag,
    )

    print("EDA completed successfully.")