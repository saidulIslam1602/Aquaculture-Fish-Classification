# 🐟 Aquaculture Fish Classification

> **A comprehensive AI-powered solution for fish species classification in aquaculture environments**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌊 Overview

This project provides a state-of-the-art machine learning solution for automatically classifying fish species in aquaculture environments. Built with PyTorch and modern computer vision techniques, it helps aquaculture professionals make data-driven decisions to optimize fish farming operations, monitor biodiversity, and improve sustainability practices.

### 🎯 Key Features

- **High-Accuracy Classification**: Advanced CNN architectures with transfer learning
- **Multiple Model Support**: ResNet, EfficientNet, Vision Transformers, and more
- **Production-Ready API**: FastAPI-based REST API for real-time inference
- **Comprehensive Training Pipeline**: Full MLOps workflow with experiment tracking
- **Data Augmentation**: Robust preprocessing and augmentation strategies
- **Model Interpretability**: Feature visualization and prediction analysis
- **Scalable Deployment**: Docker support and cloud-ready architecture

### 🐠 Supported Fish Species

The default configuration supports 10 common aquaculture species:
- Atlantic Salmon
- Rainbow Trout
- Cod
- Sea Bass
- Sea Bream
- Turbot
- Halibut
- Tuna
- Mackerel
- Other

*Species can be easily customized in the configuration file.*

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM for training, 4GB+ for inference

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/aquaculture-fish-classifier.git
cd aquaculture-fish-classifier
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dataset Preparation

Organize your fish images in the following structure:
```
data/raw/fish_images/
├── train/
│   ├── Atlantic_Salmon/
│   ├── Rainbow_Trout/
│   ├── Cod/
│   └── ...
├── val/
│   ├── Atlantic_Salmon/
│   ├── Rainbow_Trout/
│   └── ...
└── test/
    ├── Atlantic_Salmon/
    ├── Rainbow_Trout/
    └── ...
```

### Training Your Model

**Basic Training**
```bash
python scripts/train.py --data_path data/raw/fish_images --experiment_name my_fish_classifier
```

**Advanced Training with Custom Configuration**
```bash
python scripts/train.py \
    --data_path data/raw/fish_images \
    --config configs/config.yaml \
    --experiment_name resnet50_experiment \
    --output_dir experiments
```

**Debug Mode (Quick Testing)**
```bash
python scripts/train.py --data_path data/raw/fish_images --debug
```

### API Deployment

**Start the API Server**
```bash
# Set environment variables
export MODEL_PATH="experiments/my_fish_classifier/models/best_model.pth"
export CONFIG_PATH="configs/config.yaml"

# Run the API
python -m uvicorn src.api.inference:app --host 0.0.0.0 --port 8000
```

**Test the API**
```bash
# Health check
curl http://localhost:8000/health

# Predict fish species
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/fish_image.jpg"
```

## 📁 Project Structure

```
aquaculture-fish-classifier/
├── configs/                    # Configuration files
│   └── config.yaml            # Main configuration
├── data/                      # Data directory
│   ├── raw/                   # Raw image data
│   └── processed/             # Processed datasets
├── src/                       # Source code
│   ├── data/                  # Data handling modules
│   │   ├── dataset.py         # Dataset classes
│   │   └── __init__.py
│   ├── models/                # Model architectures
│   │   ├── fish_classifier.py # Main classifier
│   │   └── __init__.py
│   ├── training/              # Training pipeline
│   │   └── trainer.py         # Training logic
│   ├── utils/                 # Utility functions
│   │   ├── config.py          # Configuration management
│   │   ├── device.py          # Device setup
│   │   ├── reproducibility.py # Reproducibility utils
│   │   ├── visualization.py   # Plotting functions
│   │   └── __init__.py
│   ├── api/                   # API modules
│   │   └── inference.py       # FastAPI inference server
│   └── __init__.py
├── scripts/                   # Training scripts
│   └── train.py              # Main training script
├── models/                    # Saved models
├── results/                   # Training results
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## ⚙️ Configuration

The project uses YAML configuration files for easy customization. Key settings include:

### Model Configuration
```yaml
model:
  architecture: "resnet50"      # Model architecture
  pretrained: true              # Use pretrained weights
  num_classes: 10              # Number of fish species
  dropout: 0.3                 # Dropout rate
```

### Training Configuration
```yaml
training:
  epochs: 100                  # Number of training epochs
  learning_rate: 0.001         # Learning rate
  optimizer: "adam"            # Optimizer type
  scheduler: "cosine"          # Learning rate scheduler
```

### Data Configuration
```yaml
data:
  image_size: [224, 224]       # Input image size
  batch_size: 32               # Batch size
  augmentation:                # Data augmentation settings
    horizontal_flip: 0.5
    rotation: 15
    brightness: 0.2
```

## 🔬 Model Architectures

The project supports multiple state-of-the-art architectures:

| Architecture | Parameters | ImageNet Top-1 | Inference Speed |
|-------------|------------|----------------|-----------------|
| ResNet50    | 25.6M      | 76.1%         | Fast           |
| ResNet101   | 44.5M      | 77.4%         | Medium         |
| EfficientNet-B0 | 5.3M   | 77.1%         | Fast           |
| EfficientNet-B4 | 19.3M  | 82.9%         | Medium         |
| Vision Transformer | 86M | 81.8%         | Slow           |

## 📊 Performance Monitoring

### Experiment Tracking
- **Weights & Biases**: Comprehensive experiment tracking
- **TensorBoard**: Real-time training visualization
- **Model Checkpointing**: Automatic best model saving

### Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1 score across all classes
- **Precision/Recall**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification breakdown

## 🌐 API Documentation

### Endpoints

#### Health Check
```
GET /health
```
Returns API status and model information.

#### Single Image Prediction
```
POST /predict
```
Classify a single fish image.

**Request**: Multipart form with image file
**Response**:
```json
{
  "species": "Atlantic_Salmon",
  "confidence": 0.95,
  "probabilities": {
    "Atlantic_Salmon": 0.95,
    "Rainbow_Trout": 0.03,
    "Cod": 0.02
  },
  "processing_time": 0.123
}
```

#### Batch Prediction
```
POST /predict_batch
```
Classify multiple images (max 10 per request).

#### Species List
```
GET /species
```
Get list of supported fish species.

#### Model Information
```
GET /model_info
```
Get detailed model and configuration information.

## 🧪 Advanced Usage

### Custom Model Training

Create a custom configuration file:

```yaml
# custom_config.yaml
model:
  architecture: "efficientnet_b0"
  pretrained: true
  num_classes: 15  # Your number of species

training:
  epochs: 50
  learning_rate: 0.0005
  optimizer: "adamw"

species:
  classes:
    - "Your_Species_1"
    - "Your_Species_2"
    # ... add your species
```

Train with custom config:
```bash
python scripts/train.py \
    --config custom_config.yaml \
    --data_path your_data_path \
    --experiment_name custom_experiment
```

### Model Ensemble

Train multiple models and combine predictions:

```python
from src.models.fish_classifier import EnsembleFishClassifier

# Load multiple trained models
models = [model1, model2, model3]
ensemble = EnsembleFishClassifier(models, voting="soft")

# Use ensemble for prediction
prediction = ensemble(input_image)
```

### Transfer Learning

Fine-tune on your specific dataset:

```python
from src.models.fish_classifier import create_model

# Create model with frozen backbone
config['model']['freeze_backbone'] = True
model = create_model(config)

# Train classification head first
trainer.train(train_loader, val_loader, save_dir="phase1")

# Unfreeze and fine-tune
model.unfreeze_backbone()
trainer.train(train_loader, val_loader, save_dir="phase2")
```

## 🚀 Deployment Options

### Docker Deployment
```bash
# Build Docker image
docker build -t fish-classifier .

# Run container
docker run -p 8000:8000 -v /path/to/models:/app/models fish-classifier
```

### Cloud Deployment
- **AWS SageMaker**: Production-ready inference endpoints
- **Google Cloud AI Platform**: Scalable model serving
- **Azure ML**: Enterprise machine learning platform

## 🤝 Contributing

We welcome contributions from the aquaculture and AI communities!

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests
pytest tests/

# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/
```

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📖 Documentation

For detailed documentation, visit our [GitHub Wiki](https://github.com/your-username/aquaculture-fish-classifier/wiki).

### Available Guides
- [Dataset Preparation Guide](https://github.com/your-username/aquaculture-fish-classifier/wiki/Dataset-Preparation)
- [Model Training Tutorial](https://github.com/your-username/aquaculture-fish-classifier/wiki/Training-Tutorial)
- [API Integration Guide](https://github.com/your-username/aquaculture-fish-classifier/wiki/API-Integration)
- [Deployment Guide](https://github.com/your-username/aquaculture-fish-classifier/wiki/Deployment)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Aquaculture Industry Partners** for providing domain expertise
- **Open Source Community** for the excellent tools and libraries
- **Research Contributors** for advancing computer vision in marine biology

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **Project Lead**: [Your Name](mailto:your.email@example.com)
- **Issues**: [GitHub Issues](https://github.com/your-username/aquaculture-fish-classifier/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/aquaculture-fish-classifier/discussions)

---

**Made with ❤️ for sustainable aquaculture** 