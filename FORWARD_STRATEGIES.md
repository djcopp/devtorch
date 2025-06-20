# Forward Strategy System Tutorial

The DevTorch Forward Strategy System automatically handles different model architectures without requiring manual forward function overrides. This tutorial will guide you through using this powerful feature.

## What is the Forward Strategy System?

The Forward Strategy System intelligently detects your model architecture and automatically handles the forward pass, whether you're using simple models, complex multimodal architectures, or custom designs.

**Key Benefits:**
- No need to write custom trainer subclasses
- Automatic detection of model types
- Flexible customization when needed
- Maintains clean, modular code

## Basic Usage

### Simple Models

For standard PyTorch models with a `forward(x)` method, everything works automatically:

```python
import torch.nn as nn
from devtorch import ModelTrainer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.backbone(x)

model = SimpleModel()
trainer = ModelTrainer(model=model, ...)
trainer.train(epochs=10)
```

### MultiModel Architecture

DevTorch's `MultiModel` is automatically detected and handled:

```python
from devtorch.models import MultiModel
from devtorch.models.encoders import CNNEncoder
from devtorch.models.decoders import ClassificationDecoder

encoder = CNNEncoder(input_channels=3, output_dim=512)
decoder = ClassificationDecoder(input_dim=512, num_classes=10)

model = MultiModel(
    encoders={'backbone': encoder},
    decoders={'classification': decoder}
)

trainer = ModelTrainer(model=model, ...)
trainer.train(epochs=10)
```

## Advanced Usage

### Multimodal Models

The system automatically handles complex multimodal architectures:

```python
from devtorch.models import MultiModel
from devtorch.models.encoders import TextEncoder, ImageEncoder
from devtorch.models.processors import FusionProcessor
from devtorch.models.decoders import ClassificationDecoder

text_encoder = TextEncoder(vocab_size=10000, embed_dim=256, output_dim=512)
image_encoder = ImageEncoder(input_channels=3, output_dim=512)
fusion_processor = FusionProcessor(input_dim=512, output_dim=512)
classifier = ClassificationDecoder(input_dim=512, num_classes=2)

model = MultiModel(
    encoders={
        'text': text_encoder,
        'image': image_encoder
    },
    processors={'fusion': fusion_processor},
    decoders={'classification': classifier}
)

trainer = ModelTrainer(model=model, ...)
trainer.train(epochs=10)
```

### Automatic Input Mapping

The system automatically maps your batch inputs to the correct encoders based on naming conventions:

| Input Key | Encoder Type | Expected Inputs |
|-----------|--------------|-----------------|
| `text_input_ids` + `text_attention_mask` | Text | `(input_ids, attention_mask)` |
| `input_ids` + `attention_mask` | Text | `(input_ids, attention_mask)` |
| `image_features` or `images` | Image | `(image_tensor)` |
| `spectrogram` or `audio_features` | Audio | `(audio_tensor)` |
| `features` | Tabular | `(feature_tensor)` |
| `inputs` | Time Series | `(sequence_tensor)` |

**Example batch structure:**
```python
batch = {
    'text_input_ids': torch.tensor([...]),
    'text_attention_mask': torch.tensor([...]),
    'image_features': torch.tensor([...]),
    'labels': torch.tensor([...])
}
```

## Custom Forward Functions

When you need custom logic beyond automatic detection:

### Method 1: Pass Custom Function to Trainer

```python
def custom_forward_fn(model, batch):
    """Custom forward function with specific logic."""
    
    text_features = model.encoders['text'](
        batch['text_input_ids'], 
        batch['text_attention_mask']
    )
    
    image_features = model.encoders['image'](batch['image_features'])
    
    attention_weights = torch.sigmoid(
        torch.sum(text_features * image_features, dim=-1, keepdim=True)
    )
    
    fused_features = (
        attention_weights * text_features + 
        (1 - attention_weights) * image_features
    )
    
    outputs = model.decoders['classification'](fused_features)
    return outputs

trainer = ModelTrainer(
    model=model,
    forward_fn=custom_forward_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer={'name': 'adamw', 'lr': 1e-3}
)
```

### Method 2: Register Custom Strategy

```python
from devtorch.training.forward_strategies import CustomForwardStrategy

def my_forward_logic(model, batch):
    return model(batch['inputs'])

custom_strategy = CustomForwardStrategy(forward_fn=my_forward_logic)

trainer = ModelTrainer(model=model, ...)
trainer.forward_manager.register_custom_strategy(custom_strategy)
```

## Understanding the Detection Logic

The system uses this priority order:

1. **Custom Strategy**: If you provide a custom forward function
2. **MultiModel Strategy**: For `MultiModel` architectures
3. **Standard Strategy**: For simple models with `forward(x)`

```python
def detect_strategy(model):
    if custom_strategy_provided:
        return CustomForwardStrategy()
    elif isinstance(model, MultiModel):
        return MultiModelForwardStrategy()
    else:
        return StandardForwardStrategy()
```

## Common Patterns

### Single Task Learning
```python
model = MultiModel(
    encoders={'backbone': encoder},
    decoders={'classification': decoder}
)

trainer = ModelTrainer(
    model=model,
    loss_fn={'classification': nn.CrossEntropyLoss()},
    ...
)
```

### Multi-Task Learning
```python
model = MultiModel(
    encoders={'backbone': encoder},
    decoders={
        'classification': classification_decoder,
        'regression': regression_decoder
    }
)

trainer = ModelTrainer(
    model=model,
    loss_fn={
        'classification': nn.CrossEntropyLoss(),
        'regression': nn.MSELoss()
    },
    ...
)
```

### Shared Encoder, Multiple Tasks
```python
shared_encoder = ImageEncoder(input_channels=3, output_dim=512)

model = MultiModel(
    encoders={'shared': shared_encoder},
    decoders={
        'age_prediction': RegressionDecoder(512, 1),
        'gender_classification': ClassificationDecoder(512, 2),
        'emotion_classification': ClassificationDecoder(512, 7)
    }
)
```

## Troubleshooting

### My Model Isn't Detected Correctly

If automatic detection fails, use a custom forward function:

```python
def debug_forward(model, batch):
    print(f"Model type: {type(model)}")
    print(f"Batch keys: {batch.keys()}")
    return model(batch['your_input_key'])

trainer = ModelTrainer(model=model, forward_fn=debug_forward, ...)
```

### Input Naming Issues

Ensure your batch keys match the expected naming conventions:

```python
batch = {
    'text_input_ids': ...,        # Not 'input_ids' if you have multiple modalities
    'text_attention_mask': ...,   # Not 'attention_mask'
    'image_features': ...,        # Not 'images' if using features
    'labels': ...
}
```

### Performance Optimization

For complex models, consider implementing custom forward logic for better control:

```python
def optimized_forward(model, batch):
    with torch.no_grad():
        if not model.training:
            return efficient_inference_path(model, batch)
    
    return full_forward_path(model, batch)
```

## Best Practices

1. **Use Descriptive Batch Keys**: `text_input_ids` instead of `input_ids` for multimodal models
2. **Test Automatic Detection**: Start with automatic detection before writing custom logic
3. **Keep Custom Logic Simple**: Complex forward functions can be hard to debug
4. **Document Custom Strategies**: Add comments explaining why custom logic is needed
5. **Validate Outputs**: Ensure your custom forward function returns the expected format

This system eliminates most boilerplate while maintaining full flexibility for complex scenarios. Start with automatic detection and only add custom logic when needed! 