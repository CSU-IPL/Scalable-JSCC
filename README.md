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

Recommended conda environment: `your_env`

```bash
conda activate your_env
```

Install common dependencies:

```bash
pip install torch torchvision numpy scipy pillow matplotlib
```

## Run

### Evaluate with `main.py`

```bash
python main.py --channel-type awgn --ratio 0.7
```

### Train entropy model

```bash
python Entropy_Model_Train.py
```

### Train HFM module

```bash
python HFM_Train.py
```
