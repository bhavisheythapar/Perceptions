# mothAI
This repository contains ML code for localization, tracking and fine-grained classification of moths on trap data.

## Setup python environment
Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and prepare a python environment using the following steps:

Build a new conda environment
```bash
conda create -n milamoth_ai python=3.9
```

Install cuda toolkit and pytorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install additional libraries using pip:

```bash
python3 -m pip install -r requirements.txt
```

Activate the conda environment:
```bash
conda activate milamoth_ai
```