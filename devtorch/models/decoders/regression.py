import torch
import torch.nn as nn
from ..base import Decoder


class RegressionDecoder(Decoder):
    """Simple regression decoder with optional intermediate layers."""
    
    def __init__(self, input_dim, output_dim=1, hidden_dims=None, dropout=0.1, activation='relu'):
        super().__init__(input_dim, output_dim)
        
        if hidden_dims is None:
            self.regressor = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.regressor = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        return self.regressor(x)


class BoundedRegressionDecoder(Decoder):
    """Regression decoder with bounded outputs using sigmoid or tanh."""
    
    def __init__(self, input_dim, output_dim=1, bounds=(-1, 1), hidden_dims=None, dropout=0.1):
        super().__init__(input_dim, output_dim)
        
        self.bounds = bounds
        self.use_sigmoid = bounds[0] >= 0
        
        if hidden_dims is None:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
        
        if self.use_sigmoid:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Tanh())
        
        self.regressor = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.regressor(x)
        
        if self.use_sigmoid:
            return output * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        else:
            return output * max(abs(self.bounds[0]), abs(self.bounds[1]))


class UncertaintyRegressionDecoder(Decoder):
    """Regression decoder that outputs both prediction and uncertainty."""
    
    def __init__(self, input_dim, output_dim=1, hidden_dims=None, dropout=0.1):
        super().__init__(input_dim, output_dim * 2)
        
        if hidden_dims is None:
            self.shared = nn.Identity()
            self.mean_head = nn.Linear(input_dim, output_dim)
            self.var_head = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Softplus()
            )
        else:
            shared_layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                shared_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            self.shared = nn.Sequential(*shared_layers)
            self.mean_head = nn.Linear(prev_dim, output_dim)
            self.var_head = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softplus()
            )
    
    def forward(self, x):
        shared_features = self.shared(x)
        mean = self.mean_head(shared_features)
        variance = self.var_head(shared_features) + 1e-6
        
        return {
            'mean': mean,
            'variance': variance,
            'std': torch.sqrt(variance)
        }


class QuantileRegressionDecoder(Decoder):
    """Quantile regression decoder for prediction intervals."""
    
    def __init__(self, input_dim, output_dim=1, quantiles=[0.1, 0.5, 0.9], hidden_dims=None, dropout=0.1):
        super().__init__(input_dim, output_dim * len(quantiles))
        
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        if hidden_dims is None:
            self.shared = nn.Identity()
            shared_dim = input_dim
        else:
            shared_layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                shared_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            self.shared = nn.Sequential(*shared_layers)
            shared_dim = prev_dim
        
        self.quantile_heads = nn.ModuleList([
            nn.Linear(shared_dim, output_dim) for _ in quantiles
        ])
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        outputs = {}
        for i, (quantile, head) in enumerate(zip(self.quantiles, self.quantile_heads)):
            outputs[f'quantile_{quantile}'] = head(shared_features)
        
        return outputs


class MultiTaskRegressionDecoder(Decoder):
    """Multi-task regression decoder with shared and task-specific layers."""
    
    def __init__(self, input_dim, task_dims, shared_hidden_dim=None, task_hidden_dims=None, dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension
            task_dims: Dictionary mapping task names to output dimensions
            shared_hidden_dim: Hidden dimension for shared layers
            task_hidden_dims: Dictionary mapping task names to hidden dimensions
            dropout: Dropout rate
        """
        total_output_dim = sum(task_dims.values())
        super().__init__(input_dim, total_output_dim)
        
        self.task_names = list(task_dims.keys())
        self.task_dims = task_dims
        
        if shared_hidden_dim is None:
            shared_hidden_dim = input_dim // 2
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.task_heads = nn.ModuleDict()
        for task_name, output_dim in task_dims.items():
            if task_hidden_dims and task_name in task_hidden_dims:
                task_hidden_dim = task_hidden_dims[task_name]
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(shared_hidden_dim, task_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(task_hidden_dim, output_dim)
                )
            else:
                self.task_heads[task_name] = nn.Linear(shared_hidden_dim, output_dim)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        
        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.task_heads[task_name](shared_features)
        
        return outputs


class ResidualRegressionDecoder(Decoder):
    """Regression decoder with residual connections."""
    
    def __init__(self, input_dim, output_dim=1, num_blocks=3, hidden_dim=None, dropout=0.1):
        super().__init__(input_dim, output_dim)
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.input_projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_projection(x)
        
        for block in self.blocks:
            x = block(x)
        
        return self.output_projection(x)


class ResidualBlock(nn.Module):
    """Residual block for regression decoder."""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        residual = x
        x = self.layers(x)
        x = self.norm(x + residual)
        return torch.relu(x) 