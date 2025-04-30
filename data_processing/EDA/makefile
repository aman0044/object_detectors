.ONESHELL:
SHELL := $(shell which bash)


process_train_data:
	python eda.py --json_file_path ./data/bdd100k_labels_images_train.json \
    --csv_output_path ./data/val \
    --img_base_path ./data/bdd100k_images_100k \
	--annotation_stats_flag 
    # --required_classes "all" \
    # --image_stats_flag \
    # --annotation_stats_flag 

process_val_data:
	python eda.py --json_file_path ./data/bdd100k_labels_images_val.json \
    --csv_output_path ./data/val \
    --img_base_path ./data/bdd100k_images_100k \
	--annotation_stats_flag 
    # --required_classes "all" \
    # --image_stats_flag \
    

process_eda_data:
	make process_train_data
	make process_val_data