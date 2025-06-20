"""
Uncertainty Regression Example using DevTorch

Demonstrates regression with uncertainty quantification using:
- Synthetic tabular data with heteroscedastic noise
- Dense encoder for numerical features
- Uncertainty decoder estimating both mean and variance
- Gaussian negative log-likelihood loss
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from devtorch.models import Encoder, Decoder, MultiModel
from devtorch.training import ModelTrainer
from devtorch.training.checkpoints import BestModelCheckpoint
from devtorch.utils import set_seed, get_device_info
from .data_generators import generate_tabular_regression_data

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
    """Encoder for tabular data with numerical features."""
    
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

class UncertaintyDecoder(Decoder):
    """Regression decoder that estimates both prediction and uncertainty."""
    
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
        
        # Separate heads for mean and log variance
        self.mean_head = nn.Linear(current_dim, 1)
        self.log_var_head = nn.Linear(current_dim, 1)
    
    def forward(self, x):
        mean = self.mean_head(x)
        log_var = self.log_var_head(x)
        variance = torch.exp(log_var)
        
        return {
            'mean': mean,
            'variance': variance,
            'log_var': log_var
        }

def gaussian_nll_loss(outputs, targets):
    """Gaussian negative log-likelihood loss for uncertainty estimation."""
    mean = outputs['mean']
    log_var = outputs['log_var']
    variance = torch.exp(log_var)
    loss = 0.5 * (log_var + (targets.unsqueeze(1) - mean) ** 2 / variance)
    return loss.mean()

def run_uncertainty_regression_example():
    """Run regression with uncertainty quantification."""
    print("=== Uncertainty Regression Example ===")
    
    set_seed(42)
    
    # Generate synthetic tabular data with heteroscedastic noise
    X, y = generate_tabular_regression_data(n_samples=2000, n_features=10)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = TabularDataset(X_train, y_train)
    test_dataset = TabularDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create uncertainty model
    encoder = TabularEncoder(
        input_dim=X.shape[1],
        hidden_dims=[128, 64],
        dropout=0.3
    )
    
    decoder = UncertaintyDecoder(
        input_dim=encoder.output_dim,
        hidden_dims=[32],
        dropout=0.2
    )
    
    model = MultiModel(
        encoders={'tabular': encoder},
        decoders={'uncertainty': decoder}
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Custom trainer for uncertainty loss
    class UncertaintyTrainer(ModelTrainer):
        def forward_step(self, batch):
            features = batch['features']
            encoded = self.model.encoders['tabular'](features)
            outputs = self.model.decoders['uncertainty'](encoded)
            return outputs
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    checkpoints = [BestModelCheckpoint(monitor='val_loss', mode='min')]
    
    trainer = UncertaintyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=gaussian_nll_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_strategies=checkpoints,
        save_dir="./experiments/uncertainty_regression"
    )
    
    print("Starting training...")
    trainer.train(epochs=40, validate_every=2)
    print("Training completed!")
    
    # Evaluate with uncertainty
    model.eval()
    predictions = []
    uncertainties = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            targets = batch['targets']
            
            encoded = model.encoders['tabular'](features)
            outputs = model.decoders['uncertainty'](encoded)
            
            mean = outputs['mean']
            variance = outputs['variance']
            
            predictions.extend(mean.cpu().numpy().flatten())
            uncertainties.extend(torch.sqrt(variance).cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    
    print(f"\nUncertainty Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Mean Uncertainty: {uncertainties.mean():.4f}")
    print(f"Uncertainty Range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")
    
    # Show some predictions with uncertainties
    print(f"\nSample Predictions with Uncertainties:")
    for i in range(5):
        print(f"  Actual: {actuals[i]:.3f}, Predicted: {predictions[i]:.3f} ± {uncertainties[i]:.3f}")

if __name__ == "__main__":
    print("Device info:")
    print(get_device_info())
    print()
    
    run_uncertainty_regression_example()
    
    print("\nUncertainty regression example completed!") 