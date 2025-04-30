.ONESHELL:
SHELL := $(shell which bash)

format:
	bash format.sh

Run_EDA_Docker:
	docker build -t eda:latest data_processing/EDA
	docker run -it -rm -p 8000:8000 -v "$PWD"/data:/app/data eda:latest bash