"""
California Housing Regression Example using DevTorch

Demonstrates standard regression for tabular data using:
- Dense encoder for numerical features
- Regression decoder for continuous outputs
- California housing dataset
- Feature standardization and preprocessing
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from devtorch.models import Encoder, Decoder, MultiModel
from devtorch.training import ModelTrainer
from devtorch.training.checkpoints import BestModelCheckpoint
from devtorch.utils import set_seed, get_device_info
from .data_generators import get_california_housing_data

class TabularDataset(Dataset):
    """Dataset for tabular regression data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = None
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = {'features': self.features[idx]}
        if self.targets is not None:
            item['targets'] = self.targets[idx]
        return item

class TabularEncoder(Encoder):
    """
    Encoder for tabular data with numerical features.
    Uses batch normalization and dropout for regularization.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128],
                 dropout: float = 0.3, use_batch_norm: bool = True):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = current_dim
    
    def forward(self, x):
        return self.encoder(x)

class RegressionDecoder(Decoder):
    """Standard regression decoder."""
    
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None, 
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        self.regressor = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.regressor(x)



def run_california_housing_example():
    """Run standard regression on California housing dataset."""
    print("=== California Housing Regression Example ===")
    
    set_seed(42)
    
    # Load and preprocess data
    X, y, feature_names = get_california_housing_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Create datasets and loaders
    train_dataset = TabularDataset(X_train, y_train)
    test_dataset = TabularDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    encoder = TabularEncoder(
        input_dim=X.shape[1],
        hidden_dims=[256, 128, 64],
        dropout=0.3
    )
    
    decoder = RegressionDecoder(
        input_dim=encoder.output_dim,
        hidden_dims=[32],
        dropout=0.2
    )
    
    model = MultiModel(
        encoders={'tabular': encoder},
        decoders={'price': decoder}
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    loss_fn = nn.MSELoss()
    
    checkpoints = [BestModelCheckpoint(monitor='val_loss', mode='min')]
    
    # Train model
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_strategies=checkpoints,
        save_dir="./experiments/california_housing"
    )
    
    print("Starting training...")
    trainer.train(epochs=50, validate_every=2)
    print("Training completed!")
    
    # Evaluate model
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            targets = batch['targets']
            
            encoded = model.encoders['tabular'](features)
            outputs = model.decoders['price'](encoded)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f"\nTest Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

if __name__ == "__main__":
    print("Device info:")
    print(get_device_info())
    print()
    
    run_california_housing_example()
    
    print("\nCalifornia housing regression example completed!") 