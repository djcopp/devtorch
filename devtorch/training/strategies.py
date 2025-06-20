from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Callable
import torch
import torch.nn as nn


class ForwardStrategy(ABC):
    """Base class for forward pass strategies."""
    
    @abstractmethod
    def can_handle(self, model: nn.Module) -> bool:
        """Check if this strategy can handle the given model type."""
        pass
    
    @abstractmethod
    def forward(self, model: nn.Module, batch: Any) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Execute forward pass for the model."""
        pass


class StandardForwardStrategy(ForwardStrategy):
    """Standard forward strategy for simple models that implement forward(x)."""
    
    def can_handle(self, model: nn.Module) -> bool:
        return not (hasattr(model, 'encoders') and hasattr(model, 'decoders'))
    
    def forward(self, model: nn.Module, batch: Any) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, _ = batch
        elif isinstance(batch, dict):
            inputs = batch.get('inputs', batch.get('features', next(iter(batch.values()))))
        else:
            inputs = batch
        
        return model(inputs)


class MultiModelForwardStrategy(ForwardStrategy):
    """Forward strategy for MultiModel with automatic encoder-processor-decoder chaining."""
    
    def can_handle(self, model: nn.Module) -> bool:
        return hasattr(model, 'encoders') and hasattr(model, 'decoders')
    
    def forward(self, model: nn.Module, batch: Any) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # Handle different batch formats
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, _ = batch
        elif isinstance(batch, dict):
            inputs = batch
        else:
            inputs = {'inputs': batch}
        
        # Check if this is a single-encoder model or multi-encoder model
        if len(model.encoders) == 1:
            return self._single_encoder_forward(model, inputs)
        else:
            return self._multi_encoder_forward(model, inputs)
    
    def _single_encoder_forward(self, model: nn.Module, inputs: Dict[str, Any]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Handle single encoder models."""
        encoder_name = list(model.encoders.keys())[0]
        encoder = model.encoders[encoder_name]
        
        # Extract inputs for the encoder
        encoder_inputs = self._extract_encoder_inputs(inputs, encoder_name)
        
        # Encode
        features = encoder(*encoder_inputs) if isinstance(encoder_inputs, tuple) else encoder(encoder_inputs)
        
        # Apply processors
        if hasattr(model, 'processors'):
            for processor in model.processors:
                features = processor(features)
        
        # Apply decoders
        outputs = {}
        for decoder_name, decoder in model.decoders.items():
            outputs[decoder_name] = decoder(features)
        
        # Return single output if only one decoder, otherwise return dict
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs
    
    def _multi_encoder_forward(self, model: nn.Module, inputs: Dict[str, Any]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Handle multi-encoder models (like multimodal)."""
        encoded_features = {}
        
        # Encode with each encoder
        for encoder_name, encoder in model.encoders.items():
            encoder_inputs = self._extract_encoder_inputs(inputs, encoder_name)
            encoded_features[encoder_name] = encoder(*encoder_inputs) if isinstance(encoder_inputs, tuple) else encoder(encoder_inputs)
        
        # Apply processors (assuming they handle multiple inputs)
        processed_features = encoded_features
        if hasattr(model, 'processors') and model.processors:
            if isinstance(model.processors, dict):
                # Handle dict of processors
                for processor_name, processor in model.processors.items():
                    if hasattr(processor, 'forward'):
                        # Pass all encoded features to the processor
                        processed_features = processor(*encoded_features.values())
                        break  # Assume single processor for now
            else:
                # Handle list of processors
                current_features = list(encoded_features.values())
                for processor in model.processors:
                    if len(current_features) == 1:
                        processed_features = processor(current_features[0])
                    else:
                        processed_features = processor(*current_features)
                    current_features = [processed_features]
        
        # Apply decoders
        outputs = {}
        for decoder_name, decoder in model.decoders.items():
            if isinstance(processed_features, dict):
                decoder_input = processed_features.get(decoder_name, processed_features)
            else:
                decoder_input = processed_features
            outputs[decoder_name] = decoder(decoder_input)
        
        # Return single output if only one decoder, otherwise return dict
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs
    
    def _extract_encoder_inputs(self, inputs: Dict[str, Any], encoder_name: str) -> Union[torch.Tensor, tuple]:
        """Extract appropriate inputs for a specific encoder."""
        # Try to find encoder-specific inputs
        encoder_inputs = []
        
        # Look for keys that match the encoder name
        for key, value in inputs.items():
            if encoder_name in key:
                encoder_inputs.append(value)
        
        # If no specific inputs found, try common patterns
        if not encoder_inputs:
            if encoder_name == 'text':
                # Text encoder typically needs input_ids and attention_mask
                if 'text_input_ids' in inputs and 'text_attention_mask' in inputs:
                    return (inputs['text_input_ids'], inputs['text_attention_mask'])
                elif 'input_ids' in inputs and 'attention_mask' in inputs:
                    return (inputs['input_ids'], inputs['attention_mask'])
            elif encoder_name == 'image':
                # Image encoder typically needs image features
                if 'image_features' in inputs:
                    return inputs['image_features']
                elif 'images' in inputs:
                    return inputs['images']
            elif encoder_name == 'audio':
                # Audio encoder typically needs spectrogram
                if 'spectrogram' in inputs:
                    return inputs['spectrogram']
                elif 'audio_features' in inputs:
                    return inputs['audio_features']
            elif encoder_name == 'tabular':
                # Tabular encoder typically needs features
                if 'features' in inputs:
                    return inputs['features']
            elif encoder_name == 'timeseries':
                # Time series encoder typically needs inputs
                if 'inputs' in inputs:
                    return inputs['inputs']
            
            # Fallback to generic inputs
            if 'inputs' in inputs:
                return inputs['inputs']
            elif 'features' in inputs:
                return inputs['features']
        
        return tuple(encoder_inputs) if len(encoder_inputs) > 1 else encoder_inputs[0]


class CustomForwardStrategy(ForwardStrategy):
    """Custom forward strategy that uses a user-provided forward function."""
    
    def __init__(self, forward_fn: Callable):
        self.forward_fn = forward_fn
    
    def can_handle(self, model: nn.Module) -> bool:
        return True  # Can always handle if user provides custom function
    
    def forward(self, model: nn.Module, batch: Any) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.forward_fn(batch)


class ForwardStrategyManager:
    """Manages and selects appropriate forward strategies."""
    
    def __init__(self):
        self.strategies = [
            MultiModelForwardStrategy(),
            StandardForwardStrategy(),
        ]
        self.custom_strategy = None
    
    def register_custom_strategy(self, forward_fn: Callable):
        """Register a custom forward function."""
        self.custom_strategy = CustomForwardStrategy(forward_fn)
    
    def get_strategy(self, model: nn.Module) -> ForwardStrategy:
        """Get the appropriate strategy for the given model."""
        # Priority: Custom strategy -> Specific strategies -> Standard strategy
        if self.custom_strategy:
            return self.custom_strategy
        
        for strategy in self.strategies:
            if strategy.can_handle(model):
                return strategy
        
        # Fallback to standard strategy
        return StandardForwardStrategy()
    
    def forward(self, model: nn.Module, batch: Any) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Execute forward pass using the appropriate strategy."""
        strategy = self.get_strategy(model)
        return strategy.forward(model, batch) 