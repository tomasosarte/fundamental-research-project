FROM python:3.11-slim

# Install basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install PyTorch with or without CUDA depending on environment
RUN if command -v nvidia-smi >/dev/null 2>&1; then \
        echo "CUDA GPU detected. Installing PyTorch with CUDA 12.4..." && \
        pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 ; \
    else \
        echo "No CUDA GPU detected. Installing CPU-only PyTorch..." && \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu ; \
    fi && \
    pip install --no-cache-dir -r requirements.txt --no-deps

COPY . .

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
