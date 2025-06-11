FROM python:3.12-slim

# Optionally force CUDA installation. Set USE_CUDA=true when building to
# install the CUDA-enabled PyTorch package even if no GPU is detected at
# build time.
ARG USE_CUDA=auto

# Install basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install PyTorch with or without CUDA depending on environment or
# explicit build argument. If USE_CUDA=true the CUDA-enabled wheels are
# installed regardless of GPU detection. If USE_CUDA=false the CPU-only
# wheels are installed. Otherwise GPU detection via `nvidia-smi` is used
# to decide.
RUN if [ "$USE_CUDA" = "1" ] || [ "$USE_CUDA" = "true" ]; then \
      echo "FORCE CUDA install: installing PyTorch with CUDA 12.4..." && \
      pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision torchaudio ; \
    elif [ "$USE_CUDA" = "0" ] || [ "$USE_CUDA" = "false" ]; then \
      echo "FORCE CPU install: installing CPU-only PyTorch..." && \
      pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio ; \
    else \
      if command -v nvidia-smi >/dev/null 2>&1; then \
        echo "CUDA GPU detected. Installing PyTorch for CUDA 12.4..." && \
        pip install --no-cache-dir \
          --index-url https://download.pytorch.org/whl/cu124 \
          torch torchvision torchaudio ; \
      else \
        echo "No CUDA GPU detected. Installing CPU-only PyTorch..." && \
        pip install --no-cache-dir \
          --index-url https://download.pytorch.org/whl/cpu \
          torch torchvision torchaudio ; \
      fi ; \
    fi \
 && pip install --no-cache-dir -r requirements.txt --no-deps

COPY . .

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]