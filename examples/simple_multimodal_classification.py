"""
Simple Multimodal Classification Example using DevTorch

Demonstrates multimodal fusion for classification using:
- Text encoder for processing text descriptions
- Attention-based cross-modal fusion
- Classification on combined text and "fake image" features
- Synthetic multimodal dataset
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
from .data_generators import generate_multimodal_classification_data, MULTIMODAL_CLASSES

class MultimodalDataset(Dataset):
    """Dataset for multimodal classification with text and fake image features."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # For simplicity, simulate "image features" as random vectors
        # In practice, these would come from a pre-trained image encoder
        fake_image_features = torch.randn(512)
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'text_input_ids': encoded.input_ids.squeeze(),
            'text_attention_mask': encoded.attention_mask.squeeze(),
            'image_features': fake_image_features,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TextEncoder(Encoder):
    """Simple text encoder using transformers."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", freeze: bool = True):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        self.output_dim = self.transformer.config.hidden_size
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

class ImageEncoder(Encoder):
    """Simple encoder for fake image features."""
    
    def __init__(self, input_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_dim = output_dim
    
    def forward(self, x):
        return self.encoder(x)

class CrossModalProcessor(Processor):
    """Simple cross-modal attention processor."""
    
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim * 2
    
    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor):
        text_proj = self.text_proj(text_features).unsqueeze(1)
        image_proj = self.image_proj(image_features).unsqueeze(1)
        
        # Cross-attention
        attended_text, _ = self.attention(text_proj, image_proj, image_proj)
        attended_image, _ = self.attention(image_proj, text_proj, text_proj)
        
        attended_text = self.norm(attended_text).squeeze(1)
        attended_image = self.norm(attended_image).squeeze(1)
        
        return torch.cat([attended_text, attended_image], dim=1)

class MultimodalClassificationDecoder(Decoder):
    """Classification decoder for multimodal features."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [128]):
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

def run_multimodal_classification_example():
    """Run multimodal classification example."""
    print("=== Simple Multimodal Classification Example ===")
    
    set_seed(42)
    
    # Generate synthetic multimodal data
    texts, labels = generate_multimodal_classification_data(n_samples=500)
    
    print(f"Generated {len(texts)} samples")
    print(f"Classes: {MULTIMODAL_CLASSES}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create train/validation split
    split_idx = int(0.8 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create datasets
    train_dataset = MultimodalDataset(train_texts, train_labels, tokenizer)
    val_dataset = MultimodalDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model components
    text_encoder = TextEncoder("distilbert-base-uncased", freeze=True)
    image_encoder = ImageEncoder(input_dim=512, output_dim=256)
    
    # Cross-modal processor
    processor = CrossModalProcessor(
        text_dim=text_encoder.output_dim,
        image_dim=image_encoder.output_dim,
        hidden_dim=256
    )
    
    # Classification decoder
    decoder = MultimodalClassificationDecoder(
        input_dim=processor.output_dim,
        num_classes=len(MULTIMODAL_CLASSES),
        hidden_dims=[128, 64]
    )
    
    # Create multimodal model
    model = MultiModel(
        encoders={'text': text_encoder, 'image': image_encoder},
        processors={'fusion': processor},
        decoders={'classification': decoder}
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    loss_fn = nn.CrossEntropyLoss()
    
    checkpoints = [BestModelCheckpoint(monitor='val_accuracy', mode='max')]
    
    # Create trainer - now automatically handles MultiModel without custom forward!
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_strategies=checkpoints,
        save_dir="./experiments/multimodal_classification"
    )
    
    print("Starting training...")
    trainer.train(epochs=15, validate_every=1)
    print("Training completed!")
    
    # Test inference
    model.eval()
    print("\nTesting inference on validation samples...")
    
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        outputs = trainer.forward_manager.forward(model, sample_batch)
        predictions = torch.argmax(outputs, dim=1)
        probabilities = F.softmax(outputs, dim=1)
        
        print(f"Batch size: {len(predictions)}")
        
        for i in range(min(3, len(predictions))):
            pred_class = MULTIMODAL_CLASSES[predictions[i]]
            true_class = MULTIMODAL_CLASSES[sample_batch['labels'][i]]
            confidence = probabilities[i][predictions[i]].item()
            
            print(f"Sample {i+1}:")
            print(f"  Text: {val_texts[i][:60]}...")
            print(f"  Predicted: {pred_class} (confidence: {confidence:.3f})")
            print(f"  True: {true_class}")

if __name__ == "__main__":
    print("Device info:")
    print(get_device_info())
    print()
    
    run_multimodal_classification_example()
    
    print("\nSimple multimodal classification example completed!") 