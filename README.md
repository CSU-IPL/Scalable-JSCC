# Scalable_JSCC

PyTorch code for scalable image JSCC under wireless channels (`awgn` and `rayleigh`).

This README is rewritten for the current repository state after file cleanup.

## Paper

- Title: `Scalable Deep Joint Source-Channel Coding for Multi-User Wireless Image Transmission With Diverse Bandwidth Conditions`
- Link: `https://ieeexplore.ieee.org/document/11250876`

## Current Files

- `main.py`: Main training/evaluation pipeline.
- `model.py`: Core scalable JSCC model and pruning configs.
- `Networks.py`: Building blocks (encoders/decoders/HFM/etc.).
- `channel.py`: Channel simulation utilities.
- `datasets.py`, `Get_datasets.py`: Dataset loaders.
- `Entropy_Model_Train.py`: Entropy model training.
- `HFM_Train.py`: High-frequency module training.
- `pruning.py`: Pruning-related scripts.
- `utils.py`: CLI arguments and helper functions.

## Environment

Recommended conda environment: `wf_env`

```bash
conda activate wf_env
```

Install common dependencies:

```bash
pip install torch torchvision numpy scipy pillow matplotlib
```

## Extra Dependency (Important)

`Networks.py` and `Entropy_Model_Train.py` import modules from `src.models.*`.

You need a compatible `src/` package in this repository (or in `PYTHONPATH`) that provides at least:

- `src.models.gdn`
- `src.models.image_entropy_models`
- `src.models.layers`

Without these modules, training/inference will fail at import time.

## Dataset Layout

`main.py` reads multi-resolution data via `datasets.py` using this pattern:

```text
datasets/
  <train_dataset>/
    train/
      <R>
      <R/2>
      <R/4>
  <test_dataset>/
    val/
      <R>
      <R/2>
      <R/4>
```

Default argument values in `utils.py` are:

- `train_dataset = CelebA`
- `test_dataset = Urban`
- `train_resultion = 128`
- `test_resultion = 512`

## Model Checkpoints

`model.py` expects pretrained weights under paths like:

- `models/entropy model(no channel)/Entropy_model_64.pth`
- `models/entropy model(no channel)/Entropy_model_128.pth`
- `models/entropy model(no channel)/Entropy_model_32.pth`
- `models/awgn/random_HFM*.pth` or `models/slow rayleigh/random_HFM*.pth`

`main.py` also loads `args.save_path` when `args.load_model = True`.

## Run

### Evaluate with `main.py`

```bash
python main.py --channel-type awgn --ratio 0.7
```

or

```bash
python main.py --channel-type rayleigh --ratio 0.7
```

### Train entropy model

```bash
python Entropy_Model_Train.py
```

### Train HFM module

```bash
python HFM_Train.py
```
