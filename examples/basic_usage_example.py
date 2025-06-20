"""
Basic usage example for DevTorch showing the new components.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import DevTorch components
from devtorch import ModelTrainer
from devtorch.models import MultiModel
from devtorch.models.encoders import CNNEncoder, TextEncoder
from devtorch.models.decoders import ClassificationDecoder, RegressionDecoder
from devtorch.models.processors import FusionProcessor, AttentionProcessor


def create_dummy_data():
    """Create dummy data for demonstration."""
    
    # Image data (batch_size=100, channels=3, height=32, width=32)
    image_data = torch.randn(100, 3, 32, 32)
    
    # Text data (batch_size=100, seq_len=20) - token ids
    text_data = torch.randint(1, 1000, (100, 20))
    
    # Labels for classification (100 samples, 10 classes)
    labels = torch.randint(0, 10, (100,))
    
    return image_data, text_data, labels


def example_1_simple_image_classification():
    """Example 1: Simple image classification with CNN encoder."""
    
    print("=" * 50)
    print("Example 1: Simple Image Classification")
    print("=" * 50)
    
    # Create dummy data
    image_data, _, labels = create_dummy_data()
    
    # Create dataset and dataloader
    dataset = TensorDataset(image_data, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Create model components
    encoder = CNNEncoder(input_channels=3, output_dim=512, num_blocks=3)
    decoder = ClassificationDecoder(input_dim=512, num_classes=10)
    
    # Combine into MultiModel
    model = MultiModel(
        encoders={'backbone': encoder},
        decoders={'classification': decoder}
    )
    
    # Setup trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn={'classification': nn.CrossEntropyLoss()},
        optimizer={'name': 'adam', 'lr': 1e-3}
    )
    
    print(f"Model created successfully!")
    print(f"Encoder output dim: {encoder.output_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        outputs = model({'images': sample_batch[0]})
        print(f"Output shape: {outputs['classification'].shape}")
    
    # Train for a few epochs
    print("Training for 2 epochs...")
    trainer.train(epochs=2)
    
    print("Example 1 completed successfully!\n")


def example_2_text_classification():
    """Example 2: Text classification with LSTM encoder."""
    
    print("=" * 50)
    print("Example 2: Text Classification")
    print("=" * 50)
    
    # Create dummy data
    _, text_data, labels = create_dummy_data()
    
    # Create attention masks (assume all tokens are valid for simplicity)
    attention_masks = torch.ones_like(text_data)
    
    # Create dataset
    dataset = TensorDataset(text_data, attention_masks, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Create model components
    encoder = TextEncoder(
        vocab_size=1000, 
        embed_dim=128, 
        hidden_dim=256, 
        output_dim=512,
        num_layers=2
    )
    decoder = ClassificationDecoder(input_dim=512, num_classes=10)
    
    # Combine into MultiModel
    model = MultiModel(
        encoders={'text': encoder},
        decoders={'classification': decoder}
    )
    
    # Custom forward function for text data
    def text_forward(model, batch):
        text_ids, attention_mask, _ = batch
        text_features = model.encoders['text'](text_ids, attention_mask)
        outputs = model.decoders['classification'](text_features)
        return {'classification': outputs}
    
    # Setup trainer with custom forward
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn={'classification': nn.CrossEntropyLoss()},
        optimizer={'name': 'adam', 'lr': 1e-3},
        forward_fn=text_forward
    )
    
    print(f"Text model created successfully!")
    print(f"Encoder output dim: {encoder.output_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        outputs = text_forward(model, sample_batch)
        print(f"Output shape: {outputs['classification'].shape}")
    
    # Train for a few epochs
    print("Training for 2 epochs...")
    trainer.train(epochs=2)
    
    print("Example 2 completed successfully!\n")


def example_3_multimodal_with_fusion():
    """Example 3: Multimodal classification with fusion processor."""
    
    print("=" * 50)
    print("Example 3: Multimodal Classification with Fusion")
    print("=" * 50)
    
    # Create dummy data
    image_data, text_data, labels = create_dummy_data()
    attention_masks = torch.ones_like(text_data)
    
    # Create dataset
    dataset = TensorDataset(image_data, text_data, attention_masks, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Create model components
    image_encoder = CNNEncoder(input_channels=3, output_dim=256, num_blocks=2)
    text_encoder = TextEncoder(vocab_size=1000, embed_dim=128, output_dim=256)
    
    # Fusion processor to combine image and text features
    fusion_processor = FusionProcessor(input_dim=512, output_dim=512)  # 256 + 256 = 512
    
    # Classification decoder
    classifier = ClassificationDecoder(input_dim=512, num_classes=10)
    
    # Combine into MultiModel
    model = MultiModel(
        encoders={'image': image_encoder, 'text': text_encoder},
        processors={'fusion': fusion_processor}, 
        decoders={'classification': classifier}
    )
    
    # Custom forward function for multimodal data
    def multimodal_forward(model, batch):
        images, text_ids, attention_mask, _ = batch
        
        # Encode each modality
        image_features = model.encoders['image'](images)
        text_features = model.encoders['text'](text_ids, attention_mask)
        
        # Fuse features
        fused_features = model.processors['fusion'](image_features, text_features)
        
        # Classify
        outputs = model.decoders['classification'](fused_features)
        return {'classification': outputs}
    
    # Setup trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn={'classification': nn.CrossEntropyLoss()},
        optimizer={'name': 'adam', 'lr': 1e-3},
        forward_fn=multimodal_forward
    )
    
    print(f"Multimodal model created successfully!")
    print(f"Image encoder output dim: {image_encoder.output_dim}")
    print(f"Text encoder output dim: {text_encoder.output_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        outputs = multimodal_forward(model, sample_batch)
        print(f"Output shape: {outputs['classification'].shape}")
    
    # Train for a few epochs
    print("Training for 2 epochs...")
    trainer.train(epochs=2)
    
    print("Example 3 completed successfully!\n")


def example_4_multi_task_learning():
    """Example 4: Multi-task learning with shared encoder."""
    
    print("=" * 50)
    print("Example 4: Multi-Task Learning")
    print("=" * 50)
    
    # Create dummy data
    image_data, _, class_labels = create_dummy_data()
    regression_targets = torch.randn(100, 1)  # Continuous targets
    
    # Create dataset
    dataset = TensorDataset(image_data, class_labels, regression_targets)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Create model components
    shared_encoder = CNNEncoder(input_channels=3, output_dim=512, num_blocks=3)
    
    # Multiple decoders for different tasks
    classification_decoder = ClassificationDecoder(input_dim=512, num_classes=10)
    regression_decoder = RegressionDecoder(input_dim=512, output_dim=1)
    
    # Combine into MultiModel
    model = MultiModel(
        encoders={'shared': shared_encoder},
        decoders={
            'classification': classification_decoder,
            'regression': regression_decoder
        }
    )
    
    # Custom forward for multi-task
    def multitask_forward(model, batch):
        images, _, _ = batch
        features = model.encoders['shared'](images)
        
        return {
            'classification': model.decoders['classification'](features),
            'regression': model.decoders['regression'](features)
        }
    
    # Setup trainer with multiple loss functions
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn={
            'classification': nn.CrossEntropyLoss(),
            'regression': nn.MSELoss()
        },
        optimizer={'name': 'adam', 'lr': 1e-3},
        forward_fn=multitask_forward
    )
    
    print(f"Multi-task model created successfully!")
    print(f"Shared encoder output dim: {shared_encoder.output_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        outputs = multitask_forward(model, sample_batch)
        print(f"Classification output shape: {outputs['classification'].shape}")
        print(f"Regression output shape: {outputs['regression'].shape}")
    
    # Train for a few epochs
    print("Training for 2 epochs...")
    trainer.train(epochs=2)
    
    print("Example 4 completed successfully!\n")


if __name__ == "__main__":
    print("DevTorch Basic Usage Examples")
    print("============================\n")
    
    try:
        example_1_simple_image_classification()
        example_2_text_classification()
        example_3_multimodal_with_fusion()
        example_4_multi_task_learning()
        
        print("🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Check out HOW_TO_USE.md for detailed documentation")
        print("2. Read FORWARD_STRATEGIES.md for advanced model architectures")
        print("3. Explore other examples in the examples/ directory")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc() 