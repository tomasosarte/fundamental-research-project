# DSAIT4205 Reproduction Project: Attentive Group Equivariant Convolutional Networks

1. ## Create virtual environment ans activate it

```
python -m venv .venv
source .venv/bin/activate
```

2. ## Install requirements

```
make install
```

3. ## Experiments
Experiments are in Jupyter notebook in the `experiments` folder

4. ## Using Docker

Build the image:

```bash
docker build -t fundamental-research .
```

Run Jupyter Lab:

```bash
docker run -p 8888:8888 fundamental-research
```
