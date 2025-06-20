"""
Basic Image Classification Example using DevTorch

This example demonstrates how to use DevTorch for a simple image classification task.
It shows the core concepts: Encoders, Decoders, ModelTrainer, and all the key features.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# DevTorch imports
from devtorch import (
    SimpleEncoder, ClassificationDecoder, MultiModel,
    ModelTrainer, FolderDataset, 
    BestModelCheckpoint, LatestModelCheckpoint,
    ONNXExporter, TorchScriptExporter,
    DevLogger, set_seed, get_device
)
# Transform imports removed - using PyTorch transforms directly
from devtorch.data.transforms import accuracy


def create_model(num_classes: int = 10) -> MultiModel:
    """Create a simple classification model using DevTorch components."""
    
    # Create encoder (feature extractor)
    encoder = SimpleEncoder(
        input_channels=3,
        hidden_dims=[64, 128, 256],
        output_dim=512
    )
    
    # Create decoder (classifier)
    decoder = ClassificationDecoder(
        input_dim=512,
        num_classes=num_classes,
        dropout_rate=0.1
    )
    
    # Combine into a multi-head model (even though we only have one head)
    model = MultiModel(
        encoder=encoder,
        decoders={'classification': decoder}
    )
    
    return model


def create_datasets(data_dir: str, batch_size: int = 32):
    """Create training and validation datasets."""
    
    # Create transforms directly with PyTorch
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    # Create datasets
    # Note: This assumes your data is organized in folders like:
    # data_dir/
    #   ├── train/
    #   │   ├── class1/
    #   │   └── class2/
    #   └── val/
    #       ├── class1/
    #       └── class2/
    
    train_dataset = FolderDataset(
        root_dir=f"{data_dir}/train",
        transform=train_transform
    )
    
    val_dataset = FolderDataset(
        root_dir=f"{data_dir}/val", 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Configuration
    config = {
        'data_dir': './data',  # Change this to your data directory
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 50,
        'num_classes': 10,  # Change this to match your dataset
        'save_dir': './training_output'
    }
    
    print("DevTorch Basic Classification Example")
    print("=" * 50)
    
    # Create model
    print("Creating model...")
    model = create_model(num_classes=config['num_classes'])
    
    # Print model summary
    from devtorch.utils import print_model_summary
    print_model_summary(model, input_size=(1, 3, 224, 224))
    
    # Create datasets and data loaders
    print("Loading datasets...")
    try:
        train_loader, val_loader = create_datasets(
            config['data_dir'], 
            config['batch_size']
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except FileNotFoundError:
        print("Data directory not found. Creating dummy data loaders for demonstration.")
        # Create dummy data for demonstration
        dummy_data = [(torch.randn(3, 224, 224), torch.randint(0, config['num_classes'], (1,)).item()) 
                     for _ in range(100)]
        
        from devtorch.data import SimpleDataset
        train_dataset = SimpleDataset(dummy_data, transform=transforms.ToTensor())
        val_dataset = SimpleDataset(dummy_data[:20], transform=transforms.ToTensor())
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Setup training components
    device = get_device()
    
    # Loss function for multi-head model
    loss_fn = {'classification': nn.CrossEntropyLoss()}
    
    # Create optimizer and scheduler directly with PyTorch objects
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    
    # Metrics
    metrics = {
        'accuracy': lambda outputs, targets: accuracy(outputs, targets)
    }
    
    # Checkpoint strategies
    checkpoint_strategies = [
        BestModelCheckpoint(
            save_dir=config['save_dir'],
            monitor_metric='val/classification_accuracy',
            mode='max',
            patience=10
        ),
        LatestModelCheckpoint(
            save_dir=config['save_dir'],
            save_every_n_epochs=5,
            keep_last_n=3
        )
    ]
    
    # Create logger
    logger = DevLogger(
        log_dir=config['save_dir'],
        experiment_name='basic_classification'
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config['save_dir'],
        checkpoint_strategies=checkpoint_strategies,
        logger=logger,
        metrics=metrics
    )
    
    # Save configuration
    from devtorch.utils import save_config
    save_config(config, f"{config['save_dir']}/config.json")
    
    # Start training
    print("Starting training...")
    trainer.train(epochs=config['epochs'])
    
    # Export models for deployment
    print("Exporting models...")
    
    # Create example input for export
    example_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    onnx_exporter = ONNXExporter(
        model=model,
        export_dir=f"{config['save_dir']}/exports",
        model_name="classification_model"
    )
    
    onnx_path = onnx_exporter.export(
        example_input=example_input,
        validate=True,
        generate_usage_code=True
    )
    
    # Export to TorchScript
    torchscript_exporter = TorchScriptExporter(
        model=model,
        export_dir=f"{config['save_dir']}/exports",
        model_name="classification_model"
    )
    
    traced_path, scripted_path = torchscript_exporter.export_both(
        example_input=example_input,
        validate=True,
        generate_usage_code=True
    )
    
    print("Training and export completed!")
    print(f"Results saved to: {config['save_dir']}")
    print("\nExported models:")
    print(f"  ONNX: {onnx_path}")
    if traced_path:
        print(f"  TorchScript (traced): {traced_path}")
    if scripted_path:
        print(f"  TorchScript (scripted): {scripted_path}")


def inference_example():
    """Example of how to load and use a trained model for inference."""
    
    print("\nInference Example")
    print("=" * 30)
    
    # Load a saved model (this would be a real checkpoint in practice)
    model = create_model(num_classes=10)
    
    # Create example input
    example_input = torch.randn(1, 3, 224, 224)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(example_input)
    
    # For multi-head model, outputs is a dictionary
    classification_output = outputs['classification']
    
    # Get predictions
    _, predicted_class = torch.max(classification_output, 1)
    confidence = torch.softmax(classification_output, dim=1).max()
    
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Confidence: {confidence.item():.4f}")


if __name__ == "__main__":
    main()
    inference_example() 