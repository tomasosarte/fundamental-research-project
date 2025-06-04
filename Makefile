# Datasets
.PHONY: download_pcam

download_pcam:
	bash src/datasets/download_camelyon.sh

# Installation
.PHONY: install
install:
	pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
	pip install -r requirements.txt --no-deps