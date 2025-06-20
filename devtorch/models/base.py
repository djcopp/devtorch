from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class Encoder(nn.Module, ABC):
    """
    Base class for all encoders.
    
    Encoders transform raw input data into meaningful representations/features.
    Examples: CNN backbones, text encoders, audio feature extractors.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input into encoded representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded representation tensor
        """
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> Union[int, Tuple[int, ...]]:
        """Return the output dimension(s) of this encoder."""
        pass


class Decoder(nn.Module, ABC):
    """
    Base class for all decoders.
    
    Decoders transform encoded representations into final outputs.
    Examples: classification heads, regression outputs, segmentation heads.
    """
    
    def __init__(self, input_dim: Union[int, Tuple[int, ...]], output_dim: Union[int, Tuple[int, ...]]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform encoded representation into final output.
        
        Args:
            x: Encoded representation tensor
            
        Returns:
            Final output tensor
        """
        pass


class Processor(nn.Module, ABC):
    """
    Base class for processors.
    
    Processors are optional intermediate blocks that can transform representations
    between encoders and decoders or chain multiple processing steps.
    Examples: attention blocks, normalization layers, feature fusion modules.
    """
    
    def __init__(self, input_dim: Union[int, Tuple[int, ...]], output_dim: Union[int, Tuple[int, ...]]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        pass


class SimpleEncoder(Encoder):
    """
    A simple CNN encoder for demonstration purposes.
    
    Args:
        input_channels: Number of input channels
        hidden_dims: List of hidden dimensions for each conv layer
        output_dim: Final output dimension after global average pooling
    """
    
    def __init__(self, input_channels: int = 3, hidden_dims: List[int] = [64, 128, 256], output_dim: int = 512):
        super().__init__()
        self._output_dim = output_dim
        
        layers = []
        in_channels = input_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            in_channels = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(in_channels, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.projection(x)
        return x
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


class ClassificationDecoder(Decoder):
    """
    Simple classification decoder.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.classifier(x)


class RegressionDecoder(Decoder): 
    """
    Simple regression decoder.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, dropout_rate: float = 0.1):
        super().__init__(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.regressor(x)


class AttentionProcessor(Processor):
    """
    Simple self-attention processor.
    
    Args:
        input_dim: Input dimension
        num_heads: Number of attention heads
        dropout_rate: Dropout rate
    """
    
    def __init__(self, input_dim: int, num_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__(input_dim, input_dim)
        assert input_dim % num_heads == 0
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume x is [batch_size, seq_len, features] or reshape if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out) 