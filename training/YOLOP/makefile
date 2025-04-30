.ONESHELL:
SHELL := $(shell which bash)

ENV_NAME  =  yolo
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

infer:
	$(CONDA_ACTIVATE) $(ENV_NAME)
	python test_onnx_det.py \
	--weight yolop-640-640.onnx \
	--img_dir /mnt/03modeling/2_Aman/data/object_detection/val/ \
	--json_output_dir /mnt/03modeling/2_Aman/data/object_detection/bdd_val_infer_result/
	

