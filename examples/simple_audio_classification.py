"""
Simple Audio Classification Example using DevTorch

Demonstrates audio classification using:
- Synthetic audio spectrogram generation
- CNN encoder for spectrogram processing
- Multi-class audio classification
- Mel-spectrogram feature extraction simulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple

from devtorch.models import Encoder, Decoder, MultiModel
from devtorch.training import ModelTrainer
from devtorch.training.checkpoints import BestModelCheckpoint
from devtorch.utils import set_seed, get_device_info
from .data_generators import generate_audio_classification_data, generate_synthetic_audio_signal, AUDIO_CLASSES

class AudioDataset(Dataset):
    """Dataset for audio classification using synthetic spectrograms."""
    
    def __init__(self, audio_files: List[str], labels: List[int], n_mels: int = 128, n_frames: int = 128):
        self.audio_files = audio_files
        self.labels = labels
        self.n_mels = n_mels
        self.n_frames = n_frames
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Generate synthetic mel-spectrogram
        # In practice, this would load and process real audio files
        spectrogram = self._generate_synthetic_spectrogram(self.labels[idx])
        
        return {
            'spectrogram': spectrogram,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
    def _generate_synthetic_spectrogram(self, class_idx: int) -> torch.Tensor:
        """Generate synthetic mel-spectrogram based on class."""
        np.random.seed(hash(self.audio_files[0]) % 2147483647)  # Use filename for consistency
        
        # Create different spectrograms for different classes
        if class_idx == 0:  # Music
            # Music-like spectrogram with harmonic structure
            base = np.random.randn(self.n_mels, self.n_frames) * 0.3
            for i in range(0, self.n_mels, 8):  # Add harmonic lines
                base[i:i+2, :] += np.random.randn(2, self.n_frames) * 0.8
            base = np.abs(base)
            
        elif class_idx == 1:  # Speech
            # Speech-like spectrogram with formant structure
            base = np.random.randn(self.n_mels, self.n_frames) * 0.2
            # Add formant regions
            base[10:30, :] += np.random.randn(20, self.n_frames) * 0.6
            base[40:60, :] += np.random.randn(20, self.n_frames) * 0.5
            base = np.abs(base)
            
        elif class_idx == 2:  # Environmental
            # Environmental sound with broad spectrum
            base = np.random.randn(self.n_mels, self.n_frames) * 0.4
            base = np.abs(base)
            
        else:  # Silence
            # Very low energy across all frequencies
            base = np.random.randn(self.n_mels, self.n_frames) * 0.1
            base = np.abs(base)
        
        return torch.tensor(base, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

class AudioCNNEncoder(Encoder):
    """CNN encoder for audio spectrogram processing."""
    
    def __init__(self, input_channels: int = 1, hidden_dims: List[int] = [32, 64, 128]):
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.2)
            ])
            in_channels = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output dimensions after conv layers
        # Assuming input size (128, 128) and 3 pooling layers
        self.feature_size = hidden_dims[-1] * (128 // (2**len(hidden_dims))) * (128 // (2**len(hidden_dims)))
        
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.output_dim = hidden_dims[-1] * 16  # 4x4 = 16
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x

class AudioClassificationDecoder(Decoder):
    """Classification decoder for audio features."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)

def run_audio_classification_example():
    """Run audio classification example using synthetic spectrograms."""
    print("=== Simple Audio Classification Example ===")
    
    set_seed(42)
    
    # Generate synthetic audio file paths and labels
    audio_files, labels = generate_audio_classification_data(n_samples=500)
    
    print(f"Generated {len(audio_files)} samples")
    print(f"Classes: {AUDIO_CLASSES}")
    
    # Create train/validation split
    split_idx = int(0.8 * len(audio_files))
    train_files, val_files = audio_files[:split_idx], audio_files[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create datasets
    train_dataset = AudioDataset(train_files, train_labels, n_mels=128, n_frames=128)
    val_dataset = AudioDataset(val_files, val_labels, n_mels=128, n_frames=128)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model components
    encoder = AudioCNNEncoder(
        input_channels=1,
        hidden_dims=[32, 64, 128]
    )
    
    decoder = AudioClassificationDecoder(
        input_dim=encoder.output_dim,
        num_classes=len(AUDIO_CLASSES),
        hidden_dims=[256, 128]
    )
    
    # Create model
    model = MultiModel(
        encoders={'audio': encoder},
        decoders={'classification': decoder}
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    loss_fn = nn.CrossEntropyLoss()
    
    checkpoints = [BestModelCheckpoint(monitor='val_accuracy', mode='max')]
    
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_strategies=checkpoints,
        save_dir="./experiments/audio_classification"
    )
    
    print("Starting training...")
    trainer.train(epochs=25, validate_every=2)
    print("Training completed!")
    
    # Test inference
    model.eval()
    print("\nTesting inference on validation samples...")
    
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        spectrograms = sample_batch['spectrogram']
        true_labels = sample_batch['labels']
        
        encoded = trainer.model.encoders['audio'](spectrograms)
        outputs = trainer.model.decoders['classification'](encoded)
        
        predictions = torch.argmax(outputs, dim=1)
        probabilities = F.softmax(outputs, dim=1)
        
        print(f"Batch size: {len(predictions)}")
        
        for i in range(min(3, len(predictions))):
            pred_class = AUDIO_CLASSES[predictions[i]]
            true_class = AUDIO_CLASSES[true_labels[i]]
            confidence = probabilities[i][predictions[i]].item()
            
            print(f"Sample {i+1}:")
            print(f"  Audio file: {val_files[i]}")
            print(f"  Spectrogram shape: {spectrograms[i].shape}")
            print(f"  Predicted: {pred_class} (confidence: {confidence:.3f})")
            print(f"  True: {true_class}")

if __name__ == "__main__":
    print("Device info:")
    print(get_device_info())
    print()
    
    run_audio_classification_example()
    
    print("\nSimple audio classification example completed!") 