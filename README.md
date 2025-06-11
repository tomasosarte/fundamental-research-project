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

Build the image:

```bash
docker build -t fundamental-research .
```

Run Jupyter Lab:

```bash
docker run -p 8888:8888 fundamental-research
```
