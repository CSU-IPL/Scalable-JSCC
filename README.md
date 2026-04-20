# Scalable_JSCC

PyTorch code for scalable image JSCC under wireless channels (`awgn` and `rayleigh`).

This README is rewritten for the current repository state after file cleanup.

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

## Key Runtime Notes

- In `main.py`, `args.is_training` and `args.load_model` are set inside code, so script behavior may ignore CLI expectations unless you edit those lines.
- In `main.py`, the Rayleigh branch references `test_h_list` but the example list is commented out. Define `test_h_list` before use.
- `utils.py` currently constrains `--save_path` to two predefined checkpoint choices.

## Suggested Open-Source Cleanup

- Add a `LICENSE` file (for example, MIT or Apache-2.0).
- Pin exact dependency versions in `requirements.txt`.
- Add a short section with expected metric format and sample output.
