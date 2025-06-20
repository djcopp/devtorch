"""
Custom Forward Function Example using DevTorch

Demonstrates how to use custom forward functions with the strategy system
when the automatic model detection isn't sufficient for complex architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Any

from devtorch.models import Encoder, Decoder, Processor, MultiModel
from devtorch.training import ModelTrainer
from devtorch.training.checkpoints import BestModelCheckpoint
from devtorch.utils import set_seed, get_device_info


class ComplexMultimodalModel(MultiModel):
    """Complex multimodal model with custom forward logic."""
    
    def __init__(self, text_encoder, image_encoder, fusion_processor, classifier):
        super().__init__(
            encoders={'text': text_encoder, 'image': image_encoder},
            processors={'fusion': fusion_processor},
            decoders={'classification': classifier}
        )
        
        # Additional complex components
        self.attention_weights = nn.Parameter(torch.ones(2))
        self.gate = nn.Linear(512, 1)
    
    def custom_forward(self, batch):
        """Custom forward with complex logic."""
        # Extract inputs
        text_inputs = (batch['text_input_ids'], batch['text_attention_mask'])
        image_inputs = batch['image_features']
        
        # Encode
        text_features = self.encoders['text'](*text_inputs)
        image_features = self.encoders['image'](image_inputs)
        
        # Apply attention weighting
        weights = F.softmax(self.attention_weights, dim=0)
        weighted_text = text_features * weights[0]
        weighted_image = image_features * weights[1]
        
        # Fusion with gating mechanism
        fused = self.processors['fusion'](weighted_text, weighted_image)
        gate_value = torch.sigmoid(self.gate(fused.mean(dim=1, keepdim=True)))
        gated_features = fused * gate_value
        
        # Classification
        outputs = self.decoders['classification'](gated_features)
        
        return outputs


def run_custom_forward_example():
    """Demonstrate custom forward function usage."""
    print("=== Custom Forward Function Example ===")
    
    set_seed(42)
    
    # Create dummy data for demonstration
    class DummyDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'text_input_ids': torch.randint(0, 1000, (32,)),
                'text_attention_mask': torch.ones(32),
                'image_features': torch.randn(512),
                'labels': torch.randint(0, 3, (1,)).item()
            }
    
    # Create data loaders
    train_dataset = DummyDataset(200)
    val_dataset = DummyDataset(50)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create model components (simplified for example)
    class SimpleTextEncoder(Encoder):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 256)
            self.linear = nn.Linear(256, 256)
            self.output_dim = 256
        
        def forward(self, input_ids, attention_mask):
            x = self.embedding(input_ids)
            x = x.mean(dim=1)  # Simple pooling
            return self.linear(x)
    
    class SimpleImageEncoder(Encoder):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 256)
            self.output_dim = 256
        
        def forward(self, x):
            return self.linear(x)
    
    class SimpleFusionProcessor(Processor):
        def __init__(self):
            super().__init__(512, 512)
            self.fusion = nn.Linear(512, 512)
        
        def forward(self, text_feat, image_feat):
            combined = torch.cat([text_feat, image_feat], dim=1)
            return self.fusion(combined)
    
    class SimpleClassifier(Decoder):
        def __init__(self):
            super().__init__(512, 3)
            self.classifier = nn.Linear(512, 3)
        
        def forward(self, x):
            return self.classifier(x)
    
    # Create complex model
    model = ComplexMultimodalModel(
        text_encoder=SimpleTextEncoder(),
        image_encoder=SimpleImageEncoder(),
        fusion_processor=SimpleFusionProcessor(),
        classifier=SimpleClassifier()
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup training with custom forward function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    checkpoints = [BestModelCheckpoint(monitor='val_loss', mode='min')]
    
    # Method 1: Using custom forward function parameter
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint_strategies=checkpoints,
        save_dir="./experiments/custom_forward",
        forward_fn=model.custom_forward  # Pass custom forward function
    )
    
    print("Training with custom forward function...")
    trainer.train(epochs=3, validate_every=1)
    print("Training completed!")
    
    # Method 2: Registering custom strategy after trainer creation
    def alternative_forward(batch):
        """Alternative custom forward logic."""
        return model.custom_forward(batch)
    
    # Create new trainer and register custom strategy
    trainer2 = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint_strategies=checkpoints,
        save_dir="./experiments/custom_forward_alt"
    )
    
    # Register custom strategy
    trainer2.forward_manager.register_custom_strategy(alternative_forward)
    
    print("\nDemonstrating alternative custom forward registration...")
    
    # Test inference with custom forward
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        outputs = trainer2.forward_manager.forward(model, sample_batch)
        predictions = torch.argmax(outputs, dim=1)
        print(f"Predictions: {predictions[:5]}")


if __name__ == "__main__":
    print("Device info:")
    print(get_device_info())
    print()
    
    run_custom_forward_example()
    
    print("\nCustom forward function example completed!") 