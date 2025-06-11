# fundamental-research-project

1. ## Create virtual environment ans activate it

```
python -m venv .venv
source .venv/bin/activate
```

2. ##  Install requirements

```
make install
```

3. ## Donwload datasets


### PCAM
```
make download_pcam
```

### Rot-MNIST
The dataset can be downloaded from: https://drive.google.com/file/d/1PcPdBOyImivBz3IMYopIizGvJOnfgXGD/view?usp=sharing

4. ## Using Docker

Build the image (add `--build-arg USE_CUDA=true` if you want to force
installation of the CUDA-enabled PyTorch packages):

```bash
docker build -t fundamental-research .
# Force CUDA build
# docker build --build-arg USE_CUDA=true -t fundamental-research .
```

Run Jupyter Lab (use `--gpus all` to expose your GPU to the container):

```bash
docker run --gpus all -p 8888:8888 fundamental-research
```
