# MeanFlow: Unofficial Implementation

This repository contains an unofficial, minimalist implementation of MeanFlow, a single-step flow matching model for image generation.

## Overview
MeanFlow introduces a principled framework for one-step generative modeling by introducing the average velocity in Flow Matching methods. 

Built on the [SiT](https://github.com/willisma/SiT/tree/main) architecture, this implementation focuses on reproducing the original paper's efficient generation capabilities.

## Reproduced ImageNet Results

| Model | Epoch | FID(NFE=1), our results| FID(NFE=1), results in paper|
|---------------|---------------|----------------|----------------|
|SiT-B/4(no cfg)| 80 |58.74|61.06, Table 1f|
|SiT-B/4(w cfg)| 80 |15.43|15.53, Table 1f|
|SiT-B/2(w cfg)| 240 |6.06|6.17, Table 2|
|SiT-L/2(w cfg)| 240 |*training*|3.84, Table 2|

We are currently working on reproducing the results from the original MeanFlow paper. For detailed results and performance metrics, please refer to the original paper: [MeanFlow](https://arxiv.org/pdf/2505.13447)

## Installation

```bash
# Clone this repository
git clone https://github.com/zhuyu-cs/MeanFlow.git
cd MeanFlow

# Install dependencies
pip install -r requirements.txt
```

## Usage
### ImageNet 256

**Preparing Data**

This implementation uses LMDB datasets with VAE-encoded latents. The data preprocessing is based on the MAR approach.

```bash
# Example dataset preparation for ImageNet
cd ./preprocess_imagenet
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    main_cache.py \
    --source_lmdb /data/ImageNet_train \
    --target_lmdb /data/train_vae_latents_lmdb \
    --img_size 256 \
    --batch_size 1024 \
    --lmdb_size_gb 400
```
Note: In the example above, we assume ImageNet has already been converted to LMDB format. The preprocessing script encodes the images using the Stable Diffusion VAE and stores the latents in a new LMDB database for efficient training.

**Training**

We provide training commands for different model sizes (B, L, XL) with optimized hyperparameters based on the original paper:

```bash

accelerate launch --multi_gpu \
    train.py \
    --exp-name "meanflow_b_4" \
    --output-dir "work_dir" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-B/4" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 80\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 3.0 \ #1.0 for no cfg
    --cfg-kappa 0.\
    --cfg-min-t 0.0\
    --cfg-max-t 1.0\
    --bootstrap-ratio 0.

accelerate launch --multi_gpu \
    train.py \
    --exp-name "meanflow_b_2" \
    --output-dir "exp" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 1.0 \
    --cfg-kappa 0.5\
    --cfg-min-t 0.0\
    --cfg-max-t 1.0\
    --bootstrap-ratio 0.

accelerate launch --multi_gpu \
    train.py \
    --exp-name "meanflow_l_2" \
    --output-dir "exp" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-L/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 0.2 \
    --cfg-kappa 0.92\
    --cfg-min-t 0.0\
    --cfg-max-t 0.8\
    --bootstrap-ratio 0.

```
Each configuration is optimized for different model sizes according to the original paper's settings.

**Sampling and Evaluation**

For sampling and computing evaluation metrics (e.g., FID), we provide a distributed evaluation script:

```bash
torchrun --nproc_per_node=8 --nnodes=1 evaluate.py \
    --ckpt "/path/to/the/weights" \
    --model "SiT-L/2" \
    --resolution 256 \
    --cfg-scale 1.0 \
    --per-proc-batch-size 128 \
    --num-fid-samples 50000 \
    --sample-dir "./fid_dir" \
    --compute-metrics \
    --num-steps 1\
    --fid-statistics-file "./fid_stats/adm_in256_stats.npz"
```
This command runs sampling on 8 GPUs to generate 50,000 images for FID calculation. The script evaluates the model using a single sampling step (num-steps=1), demonstrating MeanFlow's one-step generation capability. The FID is computed against the statistics file specified in --fid-statistics-file.

**Notes**
We currently use [sd_dvae](https://huggingface.co/stabilityai/sd-vae-ft-mse), which is not the suggested tokenizer in original paper ([flaxvae](https://huggingface.co/pcuenq/sd-vae-ft-mse-flax)).

## Acknowledgements

This implementation builds upon:
- [SiT](https://github.com/willisma/SiT/tree/main) (model architecture)
- [REPA](https://github.com/sihyun-yu/REPA/tree/main) (training pipeline)
- [MAR](https://github.com/LTH14/mar/tree/main) (data preprocessing)

## Citation
If you find this implementation useful, please cite the original paper:
```
@article{geng2025mean,
  title={Mean Flows for One-step Generative Modeling},
  author={Geng, Zhengyang and Deng, Mingyang and Bai, Xingjian and Kolter, J Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}
```
## License

[MIT License](LICENSE)
