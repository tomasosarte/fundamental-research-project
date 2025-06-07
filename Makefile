# Datasets
.PHONY: download_pcam

download_pcam:
	bash src/datasets/download_camelyon.sh

.PHONY: install
install:
	@echo "Checking CUDA availability..."
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "CUDA GPU detected. Installing PyTorch with CUDA 12.4..."; \
		pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124; \
	else \
		echo "No CUDA GPU detected. Installing CPU-only PyTorch..."; \
		pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
	fi
	@echo "Installing project requirements (without deps)..."
	pip install -r requirements.txt --no-deps