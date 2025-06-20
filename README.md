# DevTorch

A simplified PyTorch framework for easy deep learning model development, designed for ML practitioners who want clean modularity without the complexity of PyTorch Lightning.

## Overview

DevTorch provides a streamlined approach to building, training, and deploying PyTorch models with a focus on:

- **Clean Modularity**: Encoder/Decoder/Processor architecture
- **Simple Training**: Powerful `ModelTrainer` with minimal boilerplate
- **Easy Data Handling**: Simple dataset classes without complex record files
- **Flexible Checkpointing**: Multiple checkpoint strategies 
- **Unified Logging**: TensorBoard + console logging
- **Model Export**: ONNX and TorchScript export with usage code generation
- **Multi-head Support**: Easy multi-task learning

## Quick Start

Check out `HOW_TO_USE.md` for a complete tutorial on getting started with DevTorch.

### Basic Example

```python
import torch
import torch.nn as nn
from devtorch import ModelTrainer
from devtorch.models import MultiModel
from devtorch.models.encoders import CNNEncoder
from devtorch.models.decoders import ClassificationDecoder

# Create model components
encoder = CNNEncoder(input_channels=3, output_dim=512)
decoder = ClassificationDecoder(input_dim=512, num_classes=10)

# Combine into multi-head model
model = MultiModel(
    encoders={'backbone': encoder},
    decoders={'classification': decoder}
)

# Setup training
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn={'classification': nn.CrossEntropyLoss()},
    optimizer={'name': 'adamw', 'lr': 1e-3}
)

# Train the model
trainer.train(epochs=50)
```

## Key Concepts

### Architecture Components

- **Encoders**: Transform raw input data into meaningful representations
- **Decoders**: Transform encoded representations into final outputs  
- **Processors**: Optional intermediate blocks for attention, normalization, etc.
- **MultiModel**: Combines encoder with multiple decoders for multi-task learning

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# The package is designed to be used directly from this directory
# Add the parent directory to your Python path or install in development mode:
pip install -e .
```

## Features

### 🏗️ Modular Architecture

- **Encoders**: Feature extraction (CNN, transformer, etc.)
- **Decoders**: Task-specific heads (classification, regression, etc.)
- **Processors**: Intermediate processing (attention, fusion, etc.)

### 🚀 Simple Training

- Automatic device handling
- Mixed precision support (optional)
- Learning rate scheduling
- Gradient clipping
- Early stopping

### 💾 Flexible Checkpointing

- `BestModelCheckpoint`: Save best model based on metrics
- `LatestModelCheckpoint`: Save latest N checkpoints
- `MultiMetricCheckpoint`: Save based on multiple metrics

### 📊 Unified Logging

- Console logging with timestamps
- TensorBoard integration
- Automatic metric tracking
- Model graph visualization

### 📦 Easy Data Handling

- `FolderDataset`: Load from folder structure
- `ImageDataset`: Load from CSV files
- `SimpleDataset`: In-memory datasets
- Built-in transform presets

### 🔄 Model Export

- ONNX export with validation
- TorchScript tracing and scripting
- Automatic usage code generation
- Model optimization

## Available Components

### Encoders
- **Image**: `ImageEncoder`, `ResNetEncoder`, `EfficientNetEncoder`, `CNNEncoder`
- **Text**: `TextEncoder`, `BERTEncoder`, `TransformerEncoder`, `RNNEncoder`
- **Audio**: `AudioEncoder` (coming soon)
- **Tabular**: `TabularEncoder` (coming soon)

### Decoders
- **Classification**: `ClassificationDecoder`, `MultiLabelClassificationDecoder`, `HierarchicalClassificationDecoder`
- **Regression**: `RegressionDecoder`, `UncertaintyRegressionDecoder`, `QuantileRegressionDecoder`
- **Advanced**: `AttentionClassificationDecoder`, `EnsembleClassificationDecoder`, `PrototypicalClassificationDecoder`

### Processors
- **Fusion**: `FusionProcessor`, `AttentionFusionProcessor`, `BilinearFusionProcessor`, `GatedFusionProcessor`
- **Attention**: `AttentionProcessor`, `SelfAttentionProcessor`, `CrossAttentionProcessor`
- **Advanced**: `LocalAttentionProcessor`, `SparseAttentionProcessor`

## Documentation

- **`HOW_TO_USE.md`**: Complete tutorial for getting started
- **`FORWARD_STRATEGIES.md`**: Advanced guide for custom model architectures
- **`examples/`**: Complete working examples for various use cases

## Examples

The `examples/` directory contains complete working examples:

- `basic_classification.py`: Simple image classification
- `text_classification.py`: Text sentiment analysis
- `multimodal_classification.py`: Image + text classification
- `multi_task_learning.py`: Single encoder, multiple tasks
- `custom_forward_example.py`: Custom forward strategies
- And many more!

## Advanced Usage

### Multi-Task Learning

```python
# Create multiple decoders
decoders = {
    'classification': ClassificationDecoder(512, 10),
    'regression': RegressionDecoder(512, 1)
}

# Multi-head model
model = MultiModel(encoder=encoder, decoders=decoders)

# Multiple loss functions
loss_fn = {
    'classification': nn.CrossEntropyLoss(),
    'regression': nn.MSELoss()
}

trainer = ModelTrainer(model=model, loss_fn=loss_fn, ...)
```

### Custom Components

```python
from devtorch.models.base import Encoder, Decoder

class MyEncoder(Encoder):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(...)
        self._output_dim = output_dim
    
    def forward(self, x):
        return self.layers(x)
    
    @property  
    def output_dim(self):
        return self._output_dim

class MyDecoder(Decoder):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.layers = nn.Sequential(...)
    
    def forward(self, x):
        return self.layers(x)
```

### Custom Checkpoint Strategies

```python
from devtorch.training.checkpoints import CheckpointStrategy

class MyCheckpointStrategy(CheckpointStrategy):
    def should_save(self, epoch, metrics):
        # Custom logic
        return epoch % 10 == 0
    
    def get_filename(self, epoch, metrics):
        return f"custom_epoch_{epoch}.pt"
```

## Comparison with PyTorch Lightning

| Feature | DevTorch | PyTorch Lightning |
|---------|----------|-------------------|
| Learning Curve | Gentle | Steep |
| Modularity | Encoder/Decoder | LightningModule |
| Configuration | Python only | YAML + Python |
| Logging | TensorBoard + Console | Multiple backends |
| Export | ONNX + TorchScript | Plugin-based |
| Multi-task | Built-in | Manual |
| Complexity | Simple | Feature-rich |

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- TensorBoard 2.8.0+
- NumPy 1.19.0+
- Pandas 1.3.0+
- Pillow 8.0.0+

## Contributing

DevTorch is designed to be simple and focused. When contributing:

1. Keep the API minimal and intuitive
2. Maintain backward compatibility
3. Add comprehensive examples
4. Follow the existing code style
5. Write clear documentation

## License

MIT License - see LICENSE file for details.

## Design Philosophy

DevTorch follows these principles:

1. **Simplicity over Features**: Better to do fewer things well
2. **Clear Abstractions**: Encoder/Decoder concept maps to how ML practitioners think
3. **No Magic**: Everything should be explicit and understandable
4. **Practical Focus**: Optimized for getting models trained and deployed quickly
5. **Minimal Dependencies**: Only essential packages required
