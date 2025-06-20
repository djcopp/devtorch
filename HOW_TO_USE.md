# How to Use DevTorch

DevTorch is designed to make deep learning model development simple and modular. This guide will get you started quickly with the core concepts and common use cases.

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Basic Classification Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from devtorch import ModelTrainer, FolderDataset
from devtorch.models import MultiModel
from devtorch.models.encoders import CNNEncoder
from devtorch.models.decoders import ClassificationDecoder

# Create model components
encoder = CNNEncoder(input_channels=3, output_dim=512)
decoder = ClassificationDecoder(input_dim=512, num_classes=10)

# Combine into model
model = MultiModel(
    encoders={'backbone': encoder},
    decoders={'classification': decoder}
)

# Setup data
train_dataset = FolderDataset('./data/train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Setup trainer
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    loss_fn={'classification': nn.CrossEntropyLoss()},
    optimizer={'name': 'adamw', 'lr': 1e-3}
)

# Train
trainer.train(epochs=10)
```

## Core Concepts

### Architecture Components

DevTorch uses a modular architecture with three main components:

**Encoders**: Transform raw input into meaningful representations
- Text encoders for NLP tasks
- Image encoders for computer vision
- Audio encoders for audio processing
- Tabular encoders for structured data

**Decoders**: Transform encoded features into final outputs
- Classification heads
- Regression heads
- Generation heads
- Custom task-specific heads

**Processors** (Optional): Intermediate processing between encoders and decoders
- Attention mechanisms
- Feature fusion
- Normalization layers
- Custom transformations

### MultiModel Architecture

The `MultiModel` class combines these components:

```python
from devtorch.models import MultiModel

model = MultiModel(
    encoders={'backbone': encoder},           # One or more encoders
    processors={'attention': processor},      # Optional processors
    decoders={'task1': decoder1, 'task2': decoder2}  # One or more decoders
)
```

## Common Use Cases

### 1. Image Classification

```python
from devtorch.models.encoders import ResNetEncoder
from devtorch.models.decoders import ClassificationDecoder

encoder = ResNetEncoder(model_name='resnet50', pretrained=True, output_dim=512)
decoder = ClassificationDecoder(input_dim=512, num_classes=1000)

model = MultiModel(
    encoders={'backbone': encoder},
    decoders={'classification': decoder}
)
```

### 2. Text Classification

```python
from devtorch.models.encoders import BERTEncoder
from devtorch.models.decoders import ClassificationDecoder

encoder = BERTEncoder(model_name='bert-base-uncased', output_dim=768)
decoder = ClassificationDecoder(input_dim=768, num_classes=2)

model = MultiModel(
    encoders={'text': encoder},
    decoders={'sentiment': decoder}
)
```

### 3. Multimodal Learning

```python
from devtorch.models.encoders import TextEncoder, ImageEncoder
from devtorch.models.processors import FusionProcessor
from devtorch.models.decoders import ClassificationDecoder

text_encoder = TextEncoder(vocab_size=10000, embed_dim=256, output_dim=512)
image_encoder = ImageEncoder(input_channels=3, output_dim=512)
fusion = FusionProcessor(input_dim=512, output_dim=512)
decoder = ClassificationDecoder(input_dim=512, num_classes=2)

model = MultiModel(
    encoders={'text': text_encoder, 'image': image_encoder},
    processors={'fusion': fusion},
    decoders={'classification': decoder}
)
```

### 4. Multi-Task Learning

```python
shared_encoder = ImageEncoder(input_channels=3, output_dim=512)

model = MultiModel(
    encoders={'shared': shared_encoder},
    decoders={
        'classification': ClassificationDecoder(512, 10),
        'regression': RegressionDecoder(512, 1),
        'segmentation': SegmentationDecoder(512, num_classes=21)
    }
)

# Multiple loss functions
trainer = ModelTrainer(
    model=model,
    loss_fn={
        'classification': nn.CrossEntropyLoss(),
        'regression': nn.MSELoss(),
        'segmentation': nn.CrossEntropyLoss()
    },
    ...
)
```

## Data Loading

### Folder Structure Dataset

```python
from devtorch import FolderDataset

# Assumes structure: root/class_name/image.jpg
dataset = FolderDataset(
    root_dir='./data/train',
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)
```

### CSV Dataset

```python
from devtorch import ImageDataset

# CSV with columns: image_path, label
dataset = ImageDataset(
    csv_file='./data/train.csv',
    root_dir='./data/images',
    transform=transforms.ToTensor()
)
```

## Training Configuration

### Basic Training

```python
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,  # Optional
    loss_fn=nn.CrossEntropyLoss(),
    optimizer={'name': 'adamw', 'lr': 1e-3}
)
```

### Advanced Training Options

```python
from devtorch.training import BestModelCheckpoint, EarlyStopping

trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer={'name': 'adamw', 'lr': 1e-3, 'weight_decay': 1e-4},
    scheduler={'name': 'cosine', 'T_max': 50},
    checkpoint_strategies=[
        BestModelCheckpoint(
            save_dir='./checkpoints',
            monitor_metric='val/accuracy',
            mode='max'
        )
    ],
    early_stopping=EarlyStopping(patience=10, monitor='val/loss'),
    mixed_precision=True,  # Use automatic mixed precision
    gradient_clip_val=1.0,  # Gradient clipping
    log_every_n_steps=100
)
```

## Model Export

### ONNX Export

```python
from devtorch.deploy import ONNXExporter

exporter = ONNXExporter(model)
exporter.export(
    example_input=torch.randn(1, 3, 224, 224),
    save_path='./model.onnx'
)
```

### TorchScript Export

```python
from devtorch.deploy import TorchScriptExporter

exporter = TorchScriptExporter(model)
exporter.export_both(
    example_input=torch.randn(1, 3, 224, 224),
    save_dir='./exports'
)
```

## Available Components

### Encoders
- `ImageEncoder`: Generic CNN encoder
- `ResNetEncoder`: ResNet-based encoder
- `BERTEncoder`: BERT-based text encoder  
- `TextEncoder`: Simple text encoder
- `AudioEncoder`: Audio processing encoder
- `TabularEncoder`: Structured data encoder
- `TimeSeriesEncoder`: Sequential data encoder

### Decoders
- `ClassificationDecoder`: Multi-class classification
- `RegressionDecoder`: Regression tasks
- `SegmentationDecoder`: Pixel-level classification
- `GenerationDecoder`: Text/sequence generation
- `MultiTaskDecoder`: Multiple outputs

### Processors
- `FusionProcessor`: Feature fusion
- `AttentionProcessor`: Attention mechanisms
- `NormalizationProcessor`: Feature normalization
- `DropoutProcessor`: Regularization

## Next Steps

1. **Check Examples**: Look at `/examples` for complete working examples
2. **Forward Strategies**: Read `FORWARD_STRATEGIES.md` for advanced model handling
3. **Custom Components**: Create your own encoders/decoders by extending base classes
4. **Deployment**: Use export tools for production deployment

## Getting Help

- Check the README.md for design philosophy
- Look at examples in `/examples` directory
- Read component-specific documentation in source files 