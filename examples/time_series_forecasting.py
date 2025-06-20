"""
LSTM Time Series Forecasting Example using DevTorch

Demonstrates time series forecasting using:
- LSTM-based encoder for sequential data processing
- Custom time series windowing and preprocessing
- Multi-step ahead forecasting
- Synthetic time series with trend and seasonality
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from devtorch.models import Encoder, Decoder, MultiModel
from devtorch.training import ModelTrainer
from devtorch.training.checkpoints import BestModelCheckpoint
from devtorch.utils import set_seed, get_device_info
from .data_generators import generate_time_series_data

class TimeSeriesDataset(Dataset):
    """
    Custom dataset for time series forecasting.
    Creates sliding windows of historical data to predict future values.
    """
    
    def __init__(self, data: np.ndarray, window_size: int, forecast_horizon: int,
                 features: Optional[List[str]] = None, target_col: int = 0):
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col
        self.features = features or [f"feature_{i}" for i in range(data.shape[1])]
        
        # Create sliding windows
        self.windows = []
        self.targets = []
        
        for i in range(len(data) - window_size - forecast_horizon + 1):
            window = data[i:i + window_size]
            self.windows.append(window)
            
            target = data[i + window_size:i + window_size + forecast_horizon, target_col]
            self.targets.append(target)
        
        self.windows = np.array(self.windows)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return {
            'inputs': torch.tensor(self.windows[idx], dtype=torch.float32),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

class LSTMTimeSeriesEncoder(Encoder):
    """
    LSTM-based encoder for time series data.
    Captures temporal dependencies in sequential data.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        return self.dropout(hidden)

class ForecastingDecoder(Decoder):
    """
    Decoder for time series forecasting.
    Predicts multiple future timesteps.
    """
    
    def __init__(self, input_dim: int, forecast_horizon: int, 
                 hidden_dims: Optional[List[int]] = None, dropout: float = 0.2):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        
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
        
        layers.append(nn.Linear(current_dim, forecast_horizon))
        self.forecaster = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.forecaster(x)

def run_forecasting_example():
    """Run LSTM-based time series forecasting example."""
    print("=== LSTM Time Series Forecasting Example ===")
    
    set_seed(42)
    
    # Generate synthetic time series data
    data = generate_time_series_data(n_timesteps=2000, n_features=3)
    print(f"Data shape: {data.shape}")
    
    # Normalize data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Split into train/test
    train_size = int(0.8 * len(data))
    train_data = data_normalized[:train_size]
    test_data = data_normalized[train_size:]
    
    # Create datasets
    window_size = 30
    forecast_horizon = 7
    
    train_dataset = TimeSeriesDataset(train_data, window_size, forecast_horizon, target_col=0)
    test_dataset = TimeSeriesDataset(test_data, window_size, forecast_horizon, target_col=0)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model components
    encoder = LSTMTimeSeriesEncoder(
        input_dim=data.shape[1],
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    )
    
    decoder = ForecastingDecoder(
        input_dim=encoder.output_dim,
        forecast_horizon=forecast_horizon,
        hidden_dims=[128, 64],
        dropout=0.2
    )
    
    # Create model
    model = MultiModel(
        encoders={'timeseries': encoder},
        decoders={'forecast': decoder}
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    loss_fn = nn.MSELoss()
    
    # Checkpoint strategies
    checkpoints = [BestModelCheckpoint(monitor='val_loss', mode='min')]
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_strategies=checkpoints,
        save_dir="./experiments/lstm_forecasting"
    )
    
    print("Starting training...")
    trainer.train(epochs=20, validate_every=1)
    print("Training completed!")
    
    # Evaluate model
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs']
            targets = batch['targets']
            
            encoded = model.encoders['timeseries'](inputs)
            outputs = model.decoders['forecast'](encoded)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
    mse = mean_squared_error(actuals.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    
    print(f"\nTest Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    print("Device info:")
    print(get_device_info())
    print()
    
    run_forecasting_example()
    print("\nTime series forecasting example completed!") 