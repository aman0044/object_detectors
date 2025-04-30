#!/bin/bash

# Define the command for processing train data
process_train_data() {
  python eda.py --json_file_path ./data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
    --csv_output_path ./data/train \
    --img_base_path ./data/bdd100k_images_100k/bdd100k/images/100k/train \
    --annotation_stats_flag \
    --image_stats_flag
    # --required_classes "all" \
    
}

# Define the command for processing validation data
process_val_data() {
  python eda.py --json_file_path ./data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
    --csv_output_path ./data/val \
    --img_base_path ./data/bdd100k_images_100k/bdd100k/images/100k/val \
    --annotation_stats_flag \
    --image_stats_flag
    # --required_classes "all" \
    # --image_stats_flag \
    
}

# Define the command for processing both train and validation data (EDA)
process_eda_data() {
  process_train_data
  process_val_data
}

# Check if the argument is provided, if not, display help
if [[ -z "$1" ]]; then
  echo "Usage: $0 {process_train_data|process_val_data|process_eda_data}"
  exit 1
fi

# Execute based on user input
case "$1" in
  process_train_data)
    process_train_data
    ;;
  process_val_data)
    process_val_data
    ;;
  process_eda_data)
    process_eda_data
    ;;
  *)
    echo "Invalid option. Usage: $0 {process_train_data|process_val_data|process_eda_data}"
    exit 1
    ;;
esac
