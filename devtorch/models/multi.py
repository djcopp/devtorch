from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .base import Encoder, Decoder, Processor


class MultiModel(nn.Module):
    """
    Multi-head model that combines an encoder with multiple decoders.
    
    Supports optional processors between encoder and decoders for advanced architectures.
    
    Args:
        encoder: The encoder to use for feature extraction
        decoders: Dictionary mapping decoder names to decoder instances
        processors: Optional list of processors to apply between encoder and decoders
        
    Example:
        ```python
        model = MultiModel(
            encoder=SimpleEncoder(output_dim=512),
            decoders={
                'classification': ClassificationDecoder(512, 10),
                'regression': RegressionDecoder(512, 1)
            }
        )
        ```
    """
    
    def __init__(
        self, 
        encoder: Encoder, 
        decoders: Dict[str, Decoder],
        processors: Optional[List[Processor]] = None
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)
        
        if processors is not None:
            self.processors = nn.ModuleList(processors)
        else:
            self.processors = nn.ModuleList()
        
        self.decoder_names = list(decoders.keys())
        self._validate_dimensions()
    
    def _validate_dimensions(self):
        """Validate that encoder output dimensions match processor/decoder input dimensions."""
        current_dim = self.encoder.output_dim
        
        for processor in self.processors:
            if processor.input_dim != current_dim:
                raise ValueError(
                    f"Processor input dim {processor.input_dim} doesn't match "
                    f"expected input dim {current_dim}"
                )
            current_dim = processor.output_dim
        
        for name, decoder in self.decoders.items():
            if decoder.input_dim != current_dim:
                raise ValueError(
                    f"Decoder '{name}' input dim {decoder.input_dim} doesn't match "
                    f"expected input dim {current_dim}"
                )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-head model.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary mapping decoder names to their outputs
        """
        # Encode input
        features = self.encoder(x)
        
        # Apply processors sequentially
        for processor in self.processors:
            features = processor(features)
        
        # Apply all decoders to get outputs
        outputs = {}
        for name, decoder in self.decoders.items():
            outputs[name] = decoder(features)
        
        return outputs
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoded features without applying decoders.
        Useful for feature extraction or visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded features after encoder and processors
        """
        features = self.encoder(x)
        
        for processor in self.processors:
            features = processor(features)
        
        return features
    
    def forward_single_head(self, x: torch.Tensor, head_name: str) -> torch.Tensor:
        """
        Forward pass through a single decoder head.
        
        Args:
            x: Input tensor
            head_name: Name of the decoder head to use
            
        Returns:
            Output from the specified decoder
        """
        if head_name not in self.decoders:
            raise ValueError(f"Decoder '{head_name}' not found. Available decoders: {self.decoder_names}")
        
        features = self.get_features(x)
        return self.decoders[head_name](features)
    
    def add_decoder(self, name: str, decoder: Decoder):
        """
        Add a new decoder head to the model.
        
        Args:
            name: Name for the new decoder
            decoder: Decoder instance to add
        """
        if name in self.decoders:
            raise ValueError(f"Decoder '{name}' already exists")
        
        # Validate dimension compatibility
        expected_dim = self.encoder.output_dim
        for processor in self.processors:
            expected_dim = processor.output_dim
            
        if decoder.input_dim != expected_dim:
            raise ValueError(
                f"New decoder input dim {decoder.input_dim} doesn't match "
                f"expected input dim {expected_dim}"
            )
        
        self.decoders[name] = decoder
        self.decoder_names.append(name)
    
    def remove_decoder(self, name: str):
        """
        Remove a decoder head from the model.
        
        Args:
            name: Name of the decoder to remove
        """
        if name not in self.decoders:
            raise ValueError(f"Decoder '{name}' not found")
        
        del self.decoders[name]
        self.decoder_names.remove(name)
    
    def freeze_encoder(self):
        """Freeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def freeze_decoder(self, name: str):
        """Freeze a specific decoder's parameters."""
        if name not in self.decoders:
            raise ValueError(f"Decoder '{name}' not found")
        
        for param in self.decoders[name].parameters():
            param.requires_grad = False
    
    def unfreeze_decoder(self, name: str):
        """Unfreeze a specific decoder's parameters."""
        if name not in self.decoders:
            raise ValueError(f"Decoder '{name}' not found")
        
        for param in self.decoders[name].parameters():
            param.requires_grad = True


class SingleModel(nn.Module):
    """
    Simple single-head model for standard use cases.
    
    Args:
        encoder: The encoder to use for feature extraction
        decoder: The decoder to use for output generation
        processors: Optional list of processors to apply between encoder and decoder
        
    Example:
        ```python
        model = SingleModel(
            encoder=SimpleEncoder(output_dim=512),
            decoder=ClassificationDecoder(512, 10)
        )
        ```
    """
    
    def __init__(
        self, 
        encoder: Encoder, 
        decoder: Decoder,
        processors: Optional[List[Processor]] = None
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        if processors is not None:
            self.processors = nn.ModuleList(processors)
        else:
            self.processors = nn.ModuleList()
        
        self._validate_dimensions()
    
    def _validate_dimensions(self):
        """Validate that encoder output dimensions match processor/decoder input dimensions."""
        current_dim = self.encoder.output_dim
        
        for processor in self.processors:
            if processor.input_dim != current_dim:
                raise ValueError(
                    f"Processor input dim {processor.input_dim} doesn't match "
                    f"expected input dim {current_dim}"
                )
            current_dim = processor.output_dim
        
        if self.decoder.input_dim != current_dim:
            raise ValueError(
                f"Decoder input dim {self.decoder.input_dim} doesn't match "
                f"expected input dim {current_dim}"
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the single-head model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output from the decoder
        """
        # Encode input
        features = self.encoder(x)
        
        # Apply processors sequentially
        for processor in self.processors:
            features = processor(features)
        
        # Apply decoder
        return self.decoder(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoded features without applying decoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded features after encoder and processors
        """
        features = self.encoder(x)
        
        for processor in self.processors:
            features = processor(features)
        
        return features 