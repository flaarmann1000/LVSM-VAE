# PRoPE
https://www.liruilong.cn/prope/

This branch implements the Novel View Synthesis experiment (Improve LVSM on RealEstate10k Dataset) for the paper:

"PRoPE: Projective Positional Encoding for Multiview Transformers"

## Setup

```
conda create -n prope python=3.10
conda activate prope
# We use CUDA 12.4 for torch. Adjust if necessary
pip install -r requirements.txt 
# This will install two packages: prope, nvs
pip install . 
```

To make sure your setup works, you could run `pytest tests/`.

## Dataset

Checkout [`scripts/data_preprocess.py`](scripts/data_preprocess.py) for converting the RealEstate10k data into our format.

## Model Training and Testing

```
# 2 GPUs Training
bash ./scripts/nvs.sh --ray_encoding plucker --pos_enc prope --gpus "0,1"
 
# 2 GPUs Testing (with zooming in)
bash ./scripts/nvs.sh --ray_encoding plucker --pos_enc prope --gpus "0,1" --test-zoom-in "1 3 5"

# 2 GPUs Testing (with more context views)
bash ./scripts/nvs.sh --ray_encoding plucker --pos_enc prope --gpus "0,1" --test-context-views "2 4 8 16"

# 2 GPUs Testing (with rendering video)
bash ./scripts/nvs.sh --ray_encoding plucker --pos_enc prope --gpus "0,1" --test-render-video
```


