# Vision Transformer (ViT) for CIFAR-100 Classification

This repository contains an implementation of the **Vision Transformer (ViT)** architecture from scratch, trained on the **CIFAR-100** dataset using **PyTorch** and managed via **Poetry**.

## Description

ViT is a deep learning model that applies the Transformer architecture, originally designed for NLP tasks, to image data. This implementation splits each image into patches, embeds them, and processes them with a standard transformer encoder to classify the input image.

## Features

- Pure ViT implementation from scratch (no pretrained models)
- Custom patch embedding and positional encoding
- Training and evaluation on CIFAR-100

## Project Structure

```
.
├── LICENSE
├── README.md
├── experiments/
│   └── ViT_classification.pth          # Trained model checkpoint
├── poetry.lock
├── pyproject.toml                      # Poetry config
├── src/
│   └── vit-cifar-100/
│       ├── config.py                   # All configuration settings
│       ├── main.py                     # Entry point
│       ├── net.py                      # ViT model architecture
│       ├── preprocessing.py            # Data loading and transforms
│       ├── train.py                    # Training and evaluation logic
│       └── __init__.py
```

## 📦 Installation

> This project uses [Poetry](https://python-poetry.org/) for dependency management.

### 1. Clone the repository

```bash
git clone https://github.com/your_username/ViT_CIFAR-100.git
cd ViT_CIFAR-100
```

### 2. Install dependencies and activate the virtual environment

```bash
poetry install
poetry shell
```
## Training

To train the model:

```bash
poetry run python src/vit-cifar-100/main.py
```

## Dataset

The model is trained on the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which contains:
- 60,000 color images of 32x32 pixels
- 100 fine-grained classes
- 50,000 training images and 10,000 test images

Dataset is upscaled up to 64x64 pixels.

The dataset is expected to be placed in: `{path_to_data}/cifar-100-python/`.

## Evaluation

Evaluation is performed during training and early stopping is applied.

Model accuracy and loss are printed during training at intervals defined by `log_interval`.


## Weights & Biases Integration

If configured, training logs can be tracked using [Weights & Biases](https://wandb.ai/).
Settings are located in:

```python
WANDB_CONFIG = {
    "project": "ViT_CIFAR-100",
}
```

To enable it, make sure you're logged into `wandb` CLI.

## 🔧 Requirements

All dependencies are listed in `pyproject.toml`

## ✍️ Author

- **Mikhail Arzhanov**
- [GitHub](https://github.com/miklrz)
- [Telegram](https://t.me/hxastur)

## 📜 License

This project is open-source and free to use under the MIT License.
