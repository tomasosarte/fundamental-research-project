FROM python:3.12-slim

# Install basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu124 \
      torch torchvision torchaudio && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" && \
    pip install --no-cache-dir -r requirements.txt --no-deps

COPY . .

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]