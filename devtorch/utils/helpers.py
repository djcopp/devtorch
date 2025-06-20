import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Specific device string ('cpu', 'cuda', 'cuda:0', etc.) or None for auto
        
    Returns:
        PyTorch device object
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device_obj = torch.device(device)
    print(f"Using device: {device_obj}")
    
    # Print additional info for CUDA
    if device_obj.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device_obj)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(device_obj) / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(device_obj) / 1024**3:.2f} GB")
    
    return device_obj


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module, input_size: Optional[tuple] = None):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Optional input size for more detailed summary
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    non_trainable_params = total_params - trainable_params
    
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    if input_size:
        print(f"Input size: {input_size}")
        
        # Try to estimate model size
        param_size = total_params * 4  # Assuming float32
        print(f"Estimated model size: {param_size / 1024**2:.2f} MB")
    
    print("=" * 60)
    
    # Print model structure
    print("\nModel Structure:")
    print(model)


def save_config(config: Dict[str, Any], filepath: Union[str, Path]):
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the config file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        try:
            json.dumps(value)  # Test if it's serializable
            serializable_config[key] = value
        except (TypeError, ValueError):
            serializable_config[key] = str(value)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Configuration saved to {filepath}")


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration dictionary from JSON file.
    
    Args:
        filepath: Path to the config file
        
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from {filepath}")
    return config


def freeze_model(model: nn.Module):
    """
    Freeze all parameters in a model.
    
    Args:
        model: PyTorch model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Frozen all parameters in {model.__class__.__name__}")


def unfreeze_model(model: nn.Module):
    """
    Unfreeze all parameters in a model.
    
    Args:
        model: PyTorch model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True
    
    print(f"Unfrozen all parameters in {model.__class__.__name__}")


def freeze_layers(model: nn.Module, layer_names: list[str]):
    """
    Freeze specific layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    frozen_count = 0
    
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                frozen_count += 1
                break
    
    print(f"Frozen {frozen_count} parameters in layers: {layer_names}")


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate from an optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']


def calculate_model_flops(model: nn.Module, input_size: tuple) -> int:
    """
    Estimate the number of FLOPs (floating point operations) for a model.
    
    Note: This is a rough estimation and may not be accurate for all model types.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        
    Returns:
        Estimated number of FLOPs
    """
    model.eval()
    
    def conv_flop_count(input_shape, output_shape, kernel_size, groups=1):
        """Calculate FLOPs for convolution layer."""
        batch_size, in_channels, input_height, input_width = input_shape
        batch_size, out_channels, output_height, output_width = output_shape
        
        kernel_flops = kernel_size[0] * kernel_size[1] * (in_channels // groups)
        output_elements = batch_size * output_height * output_width * out_channels
        
        return kernel_flops * output_elements
    
    def linear_flop_count(input_features, output_features):
        """Calculate FLOPs for linear layer."""
        return input_features * output_features
    
    total_flops = 0
    
    def flop_count_hook(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, nn.Conv2d):
            input_shape = input[0].shape
            output_shape = output.shape
            kernel_size = module.kernel_size
            groups = module.groups
            
            flops = conv_flop_count(input_shape, output_shape, kernel_size, groups)
            total_flops += flops
        
        elif isinstance(module, nn.Linear):
            input_features = input[0].shape[-1]
            output_features = output.shape[-1]
            
            flops = linear_flop_count(input_features, output_features)
            total_flops += flops
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(flop_count_hook))
    
    # Run forward pass
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops


def check_gradients(model: nn.Module) -> Dict[str, Any]:
    """
    Check gradient statistics for debugging training issues.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with gradient statistics
    """
    grad_stats = {
        'has_gradients': False,
        'total_norm': 0.0,
        'max_grad': 0.0,
        'min_grad': 0.0,
        'nan_grads': 0,
        'zero_grads': 0,
        'layer_stats': {}
    }
    
    total_norm = 0.0
    max_grad = float('-inf')
    min_grad = float('inf')
    nan_count = 0
    zero_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats['has_gradients'] = True
            
            grad_norm = param.grad.data.norm(2).item()
            total_norm += grad_norm ** 2
            
            grad_max = param.grad.data.max().item()
            grad_min = param.grad.data.min().item()
            
            max_grad = max(max_grad, grad_max)
            min_grad = min(min_grad, grad_min)
            
            # Check for NaN gradients
            if torch.isnan(param.grad.data).any():
                nan_count += 1
            
            # Check for zero gradients
            if grad_norm == 0:
                zero_count += 1
            
            grad_stats['layer_stats'][name] = {
                'norm': grad_norm,
                'max': grad_max,
                'min': grad_min,
                'mean': param.grad.data.mean().item(),
                'std': param.grad.data.std().item()
            }
    
    grad_stats['total_norm'] = total_norm ** 0.5
    grad_stats['max_grad'] = max_grad if max_grad != float('-inf') else 0.0
    grad_stats['min_grad'] = min_grad if min_grad != float('inf') else 0.0
    grad_stats['nan_grads'] = nan_count
    grad_stats['zero_grads'] = zero_count
    
    return grad_stats


def print_gradient_stats(model: nn.Module):
    """
    Print gradient statistics for debugging.
    
    Args:
        model: PyTorch model
    """
    stats = check_gradients(model)
    
    if not stats['has_gradients']:
        print("No gradients found. Make sure to call loss.backward() first.")
        return
    
    print("\nGradient Statistics:")
    print(f"Total gradient norm: {stats['total_norm']:.6f}")
    print(f"Max gradient: {stats['max_grad']:.6f}")
    print(f"Min gradient: {stats['min_grad']:.6f}")
    print(f"Layers with NaN gradients: {stats['nan_grads']}")
    print(f"Layers with zero gradients: {stats['zero_grads']}")
    
    if stats['nan_grads'] > 0:
        print("⚠ Warning: NaN gradients detected!")
    
    if stats['zero_grads'] > 0:
        print("⚠ Warning: Zero gradients detected!")


def move_to_device(obj: Any, device: torch.device) -> Any:
    """
    Recursively move tensors in nested structures to device.
    
    Args:
        obj: Object to move (tensor, list, dict, etc.)
        device: Target device
        
    Returns:
        Object with tensors moved to device
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory statistics in MB
    """
    memory_stats = {}
    
    if torch.cuda.is_available():
        memory_stats['cuda_allocated'] = torch.cuda.memory_allocated() / 1024**2
        memory_stats['cuda_reserved'] = torch.cuda.memory_reserved() / 1024**2
        memory_stats['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**2
        memory_stats['cuda_max_reserved'] = torch.cuda.max_memory_reserved() / 1024**2
    
    return memory_stats


def clear_cuda_cache():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared") 