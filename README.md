# Aquaculture Fish Classification

A machine learning solution for fish species classification in aquaculture environments using PyTorch.

## Overview

This project provides a CNN-based fish species classifier with support for multiple architectures and a REST API for inference.

### Features

- CNN architectures: ResNet, EfficientNet, Vision Transformers
- FastAPI REST API for inference
- Training pipeline with experiment tracking
- Data augmentation and preprocessing
- Model evaluation and visualization

### Supported Species

- Atlantic Salmon, Rainbow Trout, Cod, Sea Bass, Sea Bream
- Turbot, Halibut, Tuna, Mackerel, Other

## Installation

```bash
git clone https://github.com/saidulIslam1602/Aquaculture-Fish-Classification.git
cd aquaculture-fish-classifier
pip install -r requirements.txt
```

## Dataset Structure

```
data/raw/fish_images/
├── train/
│   ├── Atlantic_Salmon/
│   ├── Rainbow_Trout/
│   └── ...
├── val/
└── test/
```

## Usage

### Training

```bash
python scripts/train.py --data_path data/raw/fish_images --experiment_name my_classifier
```

### API Server

```bash
export MODEL_PATH="path/to/model.pth"
export CONFIG_PATH="configs/config.yaml"
python -m uvicorn src.api.inference:app --host 0.0.0.0 --port 8000
```

### API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single image prediction
- `POST /predict_batch` - Batch prediction
- `GET /species` - List supported species

## Project Structure

```
├── configs/          # Configuration files
├── src/             # Source code
│   ├── data/        # Dataset handling
│   ├── models/      # Model architectures
│   ├── training/    # Training pipeline
│   ├── utils/       # Utilities
│   └── api/         # API server
├── scripts/         # Training scripts
└── requirements.txt # Dependencies
```

## Configuration

Edit `configs/config.yaml` to customize:
- Model architecture and parameters
- Training hyperparameters
- Data augmentation settings
- Species classes

## Supported Architectures

- ResNet (50, 101)
- EfficientNet (B0, B4)
- Vision Transformer

## Development

```bash
# Run evaluation
python scripts/evaluate.py --model_path path/to/model.pth

# View results
python demo_results.py
``` 