"""
Sentiment Analysis Example using DevTorch

Demonstrates text classification for sentiment analysis using:
- Transformer-based text encoder (DistilBERT)
- Custom text preprocessing and tokenization
- Three-class sentiment classification (negative, neutral, positive)
- Custom training loop for transformer models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Any

from devtorch.models import Encoder, Decoder, MultiModel
from devtorch.training import ModelTrainer
from devtorch.training.checkpoints import BestModelCheckpoint
from devtorch.utils import set_seed, get_device_info
from .data_generators import generate_text_classification_data, TEXT_SENTIMENT_CLASSES

# Custom Text Dataset
class TextDataset(Dataset):
    """
    Custom dataset for text classification tasks.
    Supports both single-label and multi-label classification.
    """
    
    def __init__(self, texts: List[str], labels: List[Any], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Custom Text Encoder using transformer
class TransformerTextEncoder(Encoder):
    """
    Text encoder using pre-trained transformer models.
    Can be used with BERT, RoBERTa, DistilBERT, etc.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 freeze_base: bool = False, dropout: float = 0.1):
        super().__init__()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Output dimension is the hidden size of the transformer
        self.output_dim = self.transformer.config.hidden_size
    
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        return self.dropout(pooled_output)

# Custom Text Classification Decoder
class TextClassificationDecoder(Decoder):
    """
    Text classification decoder with optional intermediate layers.
    """
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden_dims: Optional[List[int]] = None, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        
        layers = []
        current_dim = input_dim
        
        # Add hidden layers if specified
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)

# Sentiment Analysis Decoder (specialized case)
class SentimentDecoder(Decoder):
    """
    Specialized decoder for sentiment analysis (positive/negative/neutral).
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # 3 classes: negative, neutral, positive
        )
    
    def forward(self, x):
        return self.classifier(x)

def run_sentiment_analysis_example():
    """Run sentiment analysis using transformer-based text encoder."""
    print("=== Sentiment Analysis Example ===")
    
    set_seed(42)
    
    # Generate synthetic sentiment data
    texts, labels = generate_text_classification_data(n_samples=500)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create dataset and data loader
    dataset = TextDataset(texts, labels, tokenizer, max_length=128)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)  # Using same data for demo
    
    # Create model components
    encoder = TransformerTextEncoder("distilbert-base-uncased", freeze_base=False)
    decoder = SentimentDecoder(encoder.output_dim)
    
    # Create model
    model = MultiModel(
        encoders={'text': encoder},
        decoders={'sentiment': decoder}
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
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
        save_dir="./experiments/sentiment_analysis"
    )
    
    # Custom batch handling for transformer inputs
    def custom_train_step(batch):
        """Custom training step to handle transformer inputs properly."""
        model.train()
        
        input_ids = batch['input_ids'].to(trainer.device)
        attention_mask = batch['attention_mask'].to(trainer.device)
        labels = batch['labels'].to(trainer.device)
        
        trainer.optimizer.zero_grad()
        
        # Forward pass through encoder
        encoded = trainer.model.encoders['text'](input_ids, attention_mask)
        
        # Forward pass through decoder
        outputs = trainer.model.decoders['sentiment'](encoded)
        
        # Calculate loss
        loss = trainer.loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        trainer.optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    # Override the train_step method
    trainer.train_step = custom_train_step
    
    print("Starting training...")
    
    # Train the model
    trainer.train(epochs=3, validate_every=1)
    
    print("Training completed!")
    
    # Test inference
    model.eval()
    print("\nTesting inference on sample data...")
    
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        input_ids = sample_batch['input_ids']
        attention_mask = sample_batch['attention_mask']
        true_labels = sample_batch['labels']
        
        encoded = trainer.model.encoders['text'](input_ids, attention_mask)
        outputs = trainer.model.decoders['sentiment'](encoded)
        
        predictions = torch.argmax(outputs, dim=1)
        probabilities = F.softmax(outputs, dim=1)
        
        print(f"Batch size: {len(predictions)}")
        
        for i in range(min(3, len(predictions))):
            pred_class = TEXT_SENTIMENT_CLASSES[predictions[i]]
            true_class = TEXT_SENTIMENT_CLASSES[true_labels[i]]
            confidence = probabilities[i][predictions[i]].item()
            
            print(f"Sample {i+1}:")
            print(f"  Text: {texts[i][:60]}...")
            print(f"  Predicted: {pred_class} (confidence: {confidence:.3f})")
            print(f"  True: {true_class}")

if __name__ == "__main__":
    print("Device info:")
    print(get_device_info())
    print()
    
    run_sentiment_analysis_example()
    
    print("\nSentiment analysis example completed!") 