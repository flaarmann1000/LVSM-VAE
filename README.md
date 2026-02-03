# PRoPE

This is the official repository for the project **“LVSM-VAE – Transformer-based Novel View Synthesis in Latent Space”**.

The code is adapted from **LVSM** and **PRoPE**, based on the official *Cameras as Relative Positional Encoding* repository:  
https://github.com/liruilong940607/prope

---

## TL;DR

This project was developed as part of the TUM course **Advanced Deep Learning for Computer Vision**.  
We propose **LVSM-VAE**, a latent-space extension of LVSM that performs novel view synthesis directly in a VAE latent space. This significantly reduces inference memory and runtime while scaling efficiently to larger numbers of context views and maintaining competitive reconstruction quality. Fine-tuning with additional context views further improves robustness and performance.

---

## Dataset

1. Download the **RealEstate10KSubset** from  
   https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe  
   and place it in a folder called `data`.

2. Convert the dataset to the PRoPE/LVSM format by running:
   - `src/data/utils/gen_transforms.py`
   - `src/data/utils/data_preprocess.py`

3. Convert the data to the VAE latent format using:
   - `src/data/utils/VAE_convert.py`  
   By default, this generates the `/train` split. Adjust the scripts accordingly to process `/test`.

4. Create a small sub-scene dataset for overfitting experiments:
   - `src/data/create_overfit.py`  
   This produces datasets for both pixel space and latent space.

---

## Execution

### Overfit a Single Model

```bash
./nvs.sh --model_space VAE --ray_encoding CAMRAY --pos_enc PROPE --gpus "0" --overfit 1
```

#### Train a Single Model

To train a single model, inspect the `nvs.sh` script to see the available parameters you can configure.  
Among others, you can choose:

- `model_space`: `PX` or `VAE`
- `ray_encoding`: `PLUCKER`, `CAMRAY`, `NONE`, or `RAYMAP`
- `pos_enc`: `PROPE`, `GTA`, or `NONE`

You can start a training run with:

```bash
./nvs.sh --model_space VAE --ray_encoding PLUCKER --pos_enc NONE --gpus "0"
```

#### Training in Latent Space

When training in latent space, you can choose whether to decode images during the validation step.  
Decoding allows you to evaluate perceptual metrics but requires significantly more compute and therefore increases training time.

```bash
./nvs.sh --model_space VAE --ray_encoding PLUCKER --pos_enc NONE --gpus "0" --decode 1
```


#### Iteratively Train Multiple Combinations

To run multiple training configurations sequentially, you can use `run_multi.sh`.  
This is useful for evaluating different overfitting settings or framework backends in a single run.

```bash
./run_multi.sh \
  --gpus 0 \
  --cmd " --overfit 1 --model_space VAE --ray_encoding CAMRAY --pos_enc PROPE --max_steps 20000 --from_torch 1" \
  --cmd " --overfit 1 --model_space VAE --ray_encoding CAMRAY --pos_enc PROPE --max_steps 20000 --from_torch 0" \
  --cmd " --overfit 4 --model_space VAE --ray_encoding CAMRAY --pos_enc PROPE --max_steps 20000 --from_torch 1" \
  --cmd " --overfit 4 --model_space VAE --ray_encoding CAMRAY --pos_enc PROPE --max_steps 20000 --from_torch 0"
``` 

## Dockerfile

We created a dockerfile for easy usage on an external SSH server

1. Build Docker file
```
docker build -t lvsm .
```

2. Run it all files in workspace
```
docker run -it -v $(pwd):/workspace lvsm bash
```