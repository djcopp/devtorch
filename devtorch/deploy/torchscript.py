import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn


class TorchScriptExporter:
    """
    Export PyTorch models to TorchScript format with validation and usage code generation.
    
    Args:
        model: PyTorch model to export
        export_dir: Directory to save exported model and files
        model_name: Name for the exported model (default: model class name)
    """
    
    def __init__(
        self,
        model: nn.Module,
        export_dir: str = "./exported_models",
        model_name: Optional[str] = None
    ):
        self.model = model
        self.export_dir = Path(export_dir)
        self.model_name = model_name or model.__class__.__name__
        
        # Create export directory
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Model should be in eval mode for export
        self.model.eval()
    
    def export_trace(
        self,
        example_input: torch.Tensor,
        validate: bool = True,
        generate_usage_code: bool = True,
        optimize: bool = True
    ) -> str:
        """
        Export model using TorchScript tracing.
        
        Args:
            example_input: Example input tensor for tracing
            validate: Whether to validate the traced model
            generate_usage_code: Whether to generate usage code
            optimize: Whether to optimize the traced model
            
        Returns:
            Path to the exported TorchScript model
        """
        torchscript_path = self.export_dir / f"{self.model_name}_traced.pt"
        
        print(f"Tracing model to TorchScript: {torchscript_path}")
        
        try:
            # Trace the model
            traced_model = torch.jit.trace(self.model, example_input)
            
            # Optimize if requested
            if optimize:
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Save the traced model
            traced_model.save(str(torchscript_path))
            
            print(f"✓ Successfully traced and saved to {torchscript_path}")
            
        except Exception as e:
            print(f"✗ Tracing failed: {e}")
            raise
        
        # Validate traced model
        if validate:
            self._validate_torchscript_model(str(torchscript_path), example_input)
        
        # Generate usage code
        if generate_usage_code:
            self._generate_usage_code(torchscript_path, example_input, "trace")
        
        return str(torchscript_path)
    
    def export_script(
        self,
        validate_input: Optional[torch.Tensor] = None,
        generate_usage_code: bool = True,
        optimize: bool = True
    ) -> str:
        """
        Export model using TorchScript scripting (compilation).
        
        Args:
            validate_input: Optional input for validation
            generate_usage_code: Whether to generate usage code
            optimize: Whether to optimize the scripted model
            
        Returns:
            Path to the exported TorchScript model
        """
        torchscript_path = self.export_dir / f"{self.model_name}_scripted.pt"
        
        print(f"Scripting model to TorchScript: {torchscript_path}")
        
        try:
            # Script the model
            scripted_model = torch.jit.script(self.model)
            
            # Optimize if requested
            if optimize:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
            # Save the scripted model
            scripted_model.save(str(torchscript_path))
            
            print(f"✓ Successfully scripted and saved to {torchscript_path}")
            
        except Exception as e:
            print(f"✗ Scripting failed: {e}")
            print("Tip: Scripting requires all operations to be TorchScript compatible.")
            print("Try using tracing instead with export_trace().")
            raise
        
        # Validate scripted model
        if validate_input is not None:
            self._validate_torchscript_model(str(torchscript_path), validate_input)
        
        # Generate usage code
        if generate_usage_code:
            self._generate_usage_code(torchscript_path, validate_input, "script")
        
        return str(torchscript_path)
    
    def export_both(
        self,
        example_input: torch.Tensor,
        validate: bool = True,
        generate_usage_code: bool = True,
        optimize: bool = True
    ) -> tuple[str, str]:
        """
        Export model using both tracing and scripting.
        
        Args:
            example_input: Example input tensor
            validate: Whether to validate the models
            generate_usage_code: Whether to generate usage code
            optimize: Whether to optimize the models
            
        Returns:
            Tuple of (traced_path, scripted_path)
        """
        print("Exporting model using both tracing and scripting...")
        
        # Export traced version
        try:
            traced_path = self.export_trace(
                example_input, 
                validate=validate, 
                generate_usage_code=False,
                optimize=optimize
            )
        except Exception as e:
            print(f"Tracing failed: {e}")
            traced_path = None
        
        # Export scripted version
        try:
            scripted_path = self.export_script(
                validate_input=example_input if validate else None,
                generate_usage_code=False,
                optimize=optimize
            )
        except Exception as e:
            print(f"Scripting failed: {e}")
            scripted_path = None
        
        # Generate combined usage code
        if generate_usage_code:
            self._generate_combined_usage_code(
                traced_path, 
                scripted_path, 
                example_input
            )
        
        return traced_path, scripted_path
    
    def _validate_torchscript_model(self, torchscript_path: str, example_input: torch.Tensor):
        """Validate that the TorchScript model produces the same outputs as PyTorch."""
        print("Validating TorchScript model...")
        
        try:
            # Load TorchScript model
            loaded_model = torch.jit.load(torchscript_path)
            loaded_model.eval()
            
            # Get original PyTorch output
            with torch.no_grad():
                original_output = self.model(example_input)
            
            # Get TorchScript output
            with torch.no_grad():
                torchscript_output = loaded_model(example_input)
            
            # Compare outputs
            if isinstance(original_output, dict) and isinstance(torchscript_output, dict):
                # Multi-head model comparison
                all_close = True
                for key in original_output.keys():
                    if key in torchscript_output:
                        if not torch.allclose(original_output[key], torchscript_output[key], rtol=1e-4, atol=1e-6):
                            print(f"⚠ Output '{key}' differs between PyTorch and TorchScript")
                            all_close = False
                    else:
                        print(f"⚠ Output '{key}' missing in TorchScript model")
                        all_close = False
            
            elif isinstance(original_output, (list, tuple)) and isinstance(torchscript_output, (list, tuple)):
                # Multi-output model comparison
                all_close = True
                for i, (orig, ts) in enumerate(zip(original_output, torchscript_output)):
                    if not torch.allclose(orig, ts, rtol=1e-4, atol=1e-6):
                        print(f"⚠ Output {i} differs between PyTorch and TorchScript")
                        all_close = False
            
            else:
                # Single output comparison
                all_close = torch.allclose(original_output, torchscript_output, rtol=1e-4, atol=1e-6)
                if not all_close:
                    print("⚠ Output differs between PyTorch and TorchScript")
            
            if all_close:
                print("✓ TorchScript model validation passed")
            else:
                print("⚠ TorchScript model validation failed - outputs differ")
                
        except Exception as e:
            print(f"⚠ TorchScript validation failed: {e}")
    
    def _generate_usage_code(
        self,
        torchscript_path: Path,
        example_input: Optional[torch.Tensor],
        export_type: str
    ):
        """Generate Python code showing how to use the exported TorchScript model."""
        
        # Get input shape information
        if example_input is not None:
            if isinstance(example_input, (list, tuple)):
                input_shapes = [list(inp.shape) for inp in example_input]
            else:
                input_shapes = [list(example_input.shape)]
        else:
            input_shapes = ["Unknown - no example input provided"]
        
        usage_code = f'''"""
TorchScript Model Usage Code for {self.model_name}
Generated by DevTorch

Export method: {export_type}
This code shows how to load and use the exported TorchScript model.
"""

import torch

def load_model(model_path: str = "{torchscript_path.name}"):
    """Load the TorchScript model."""
    model = torch.jit.load(model_path)
    model.eval()
    return model

def predict(model, input_data):
    """
    Run inference on the TorchScript model.
    
    Args:
        model: Loaded TorchScript model
        input_data: Input tensor or list of tensors
    
    Returns:
        Model outputs
    """
    with torch.no_grad():
        outputs = model(input_data)
    return outputs

def example_usage():
    """Example of how to use the model."""
    # Load model
    model = load_model()
    
    # Create example input
'''

        if len(input_shapes) == 1 and input_shapes[0] != "Unknown - no example input provided":
            usage_code += f'''    example_input = torch.randn{tuple(input_shapes[0])}
    
    # Expected input shape: {input_shapes[0]}
    print(f"Input shape: {{example_input.shape}}")
    
    # Run prediction
    outputs = predict(model, example_input)
    
    # Print output information
    if isinstance(outputs, dict):
        print("Multi-head model outputs:")
        for key, value in outputs.items():
            print(f"  {{key}}: {{value.shape}}")
    elif isinstance(outputs, (list, tuple)):
        print("Multi-output model:")
        for i, output in enumerate(outputs):
            print(f"  Output {{i}}: {{output.shape}}")
    else:
        print(f"Output shape: {{outputs.shape}}")
'''
        elif len(input_shapes) > 1:
            usage_code += '''    example_inputs = [
'''
            for i, shape in enumerate(input_shapes):
                usage_code += f'''        torch.randn{tuple(shape)},  # Input {i}
'''
            usage_code += '''    ]
    
    print("Input shapes:", [inp.shape for inp in example_inputs])
    
    # Run prediction
    outputs = predict(model, example_inputs)
    
    # Print output information
    if isinstance(outputs, dict):
        print("Multi-head model outputs:")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape}")
    elif isinstance(outputs, (list, tuple)):
        print("Multi-output model:")
        for i, output in enumerate(outputs):
            print(f"  Output {i}: {output.shape}")
    else:
        print(f"Output shape: {outputs.shape}")
'''
        else:
            usage_code += '''    # Create your input tensor here
    # example_input = torch.randn(batch_size, channels, height, width)  # for images
    # example_input = torch.randn(batch_size, sequence_length, features)  # for sequences
    
    print("Please create appropriate input tensor for your model")
    return
'''

        usage_code += '''

def benchmark_model(model, input_data, num_runs: int = 100):
    """Benchmark the TorchScript model performance."""
    import time
    
    # Warmup
    for _ in range(10):
        _ = predict(model, input_data)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = predict(model, input_data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.1f} inferences/second")

if __name__ == "__main__":
    example_usage()
'''

        # Save usage code
        usage_file = self.export_dir / f"{self.model_name}_torchscript_usage.py"
        with open(usage_file, 'w') as f:
            f.write(usage_code)
        
        print(f"✓ Generated TorchScript usage code: {usage_file}")
    
    def _generate_combined_usage_code(
        self,
        traced_path: Optional[str],
        scripted_path: Optional[str],
        example_input: torch.Tensor
    ):
        """Generate usage code that can handle both traced and scripted models."""
        
        if not traced_path and not scripted_path:
            print("⚠ No models exported successfully, skipping usage code generation")
            return
        
        # Get input shape information
        if isinstance(example_input, (list, tuple)):
            input_shapes = [list(inp.shape) for inp in example_input]
        else:
            input_shapes = [list(example_input.shape)]
        
        usage_code = f'''"""
Combined TorchScript Model Usage Code for {self.model_name}
Generated by DevTorch

This code can load and use both traced and scripted versions of the model.
"""

import torch
from pathlib import Path

def load_model(model_type: str = "auto"):
    """
    Load the TorchScript model.
    
    Args:
        model_type: Type of model to load ("traced", "scripted", or "auto")
    
    Returns:
        Loaded TorchScript model
    """
'''
        
        if traced_path and scripted_path:
            usage_code += f'''    traced_path = "{Path(traced_path).name}"
    scripted_path = "{Path(scripted_path).name}"
    
    if model_type == "traced":
        if Path(traced_path).exists():
            model = torch.jit.load(traced_path)
            print(f"Loaded traced model: {{traced_path}}")
        else:
            raise FileNotFoundError(f"Traced model not found: {{traced_path}}")
    elif model_type == "scripted":
        if Path(scripted_path).exists():
            model = torch.jit.load(scripted_path)
            print(f"Loaded scripted model: {{scripted_path}}")
        else:
            raise FileNotFoundError(f"Scripted model not found: {{scripted_path}}")
    elif model_type == "auto":
        # Try traced first, then scripted
        if Path(traced_path).exists():
            model = torch.jit.load(traced_path)
            print(f"Loaded traced model: {{traced_path}}")
        elif Path(scripted_path).exists():
            model = torch.jit.load(scripted_path)
            print(f"Loaded scripted model: {{scripted_path}}")
        else:
            raise FileNotFoundError("No TorchScript models found")
    else:
        raise ValueError("model_type must be 'traced', 'scripted', or 'auto'")
'''
        elif traced_path:
            usage_code += f'''    traced_path = "{Path(traced_path).name}"
    
    if Path(traced_path).exists():
        model = torch.jit.load(traced_path)
        print(f"Loaded traced model: {{traced_path}}")
    else:
        raise FileNotFoundError(f"Traced model not found: {{traced_path}}")
'''
        elif scripted_path:
            usage_code += f'''    scripted_path = "{Path(scripted_path).name}"
    
    if Path(scripted_path).exists():
        model = torch.jit.load(scripted_path)
        print(f"Loaded scripted model: {{scripted_path}}")
    else:
        raise FileNotFoundError(f"Scripted model not found: {{scripted_path}}")
'''

        usage_code += '''
    model.eval()
    return model

def predict(model, input_data):
    """Run inference on the TorchScript model."""
    with torch.no_grad():
        outputs = model(input_data)
    return outputs

def compare_models():
    """Compare performance between traced and scripted models."""
    import time
    
'''
        
        if traced_path and scripted_path:
            usage_code += f'''    print("Loading both models for comparison...")
    traced_model = load_model("traced")
    scripted_model = load_model("scripted")
    
    # Create example input
    example_input = torch.randn{tuple(input_shapes[0])}
    
    # Warmup
    for _ in range(10):
        _ = predict(traced_model, example_input)
        _ = predict(scripted_model, example_input)
    
    # Benchmark traced model
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        _ = predict(traced_model, example_input)
    traced_time = (time.time() - start_time) / num_runs * 1000
    
    # Benchmark scripted model
    start_time = time.time()
    for _ in range(num_runs):
        _ = predict(scripted_model, example_input)
    scripted_time = (time.time() - start_time) / num_runs * 1000
    
    print(f"Traced model avg time: {{traced_time:.2f}} ms")
    print(f"Scripted model avg time: {{scripted_time:.2f}} ms")
    
    if traced_time < scripted_time:
        print("Traced model is faster")
    elif scripted_time < traced_time:
        print("Scripted model is faster")
    else:
        print("Both models have similar performance")
'''
        else:
            usage_code += '''    print("Only one model type available - cannot compare")
'''

        usage_code += f'''

def example_usage():
    """Example of how to use the model."""
    # Load model (auto-selects best available)
    model = load_model("auto")
    
    # Create example input
    example_input = torch.randn{tuple(input_shapes[0])}
    print(f"Input shape: {{example_input.shape}}")
    
    # Run prediction
    outputs = predict(model, example_input)
    
    # Print output information
    if isinstance(outputs, dict):
        print("Multi-head model outputs:")
        for key, value in outputs.items():
            print(f"  {{key}}: {{value.shape}}")
    elif isinstance(outputs, (list, tuple)):
        print("Multi-output model:")
        for i, output in enumerate(outputs):
            print(f"  Output {{i}}: {{output.shape}}")
    else:
        print(f"Output shape: {{outputs.shape}}")

if __name__ == "__main__":
    example_usage()
    
    # Uncomment to compare model performance
    # compare_models()
'''

        # Save combined usage code
        usage_file = self.export_dir / f"{self.model_name}_torchscript_combined_usage.py"
        with open(usage_file, 'w') as f:
            f.write(usage_code)
        
        print(f"✓ Generated combined TorchScript usage code: {usage_file}")
    
    def get_model_info(self, torchscript_path: str) -> dict:
        """Get information about the exported TorchScript model."""
        try:
            model = torch.jit.load(torchscript_path)
            
            info = {
                "model_type": "TorchScript",
                "file_size_mb": Path(torchscript_path).stat().st_size / (1024 * 1024),
                "graph": str(model.graph),
                "code": str(model.code) if hasattr(model, 'code') else "Not available"
            }
            
            return info
            
        except Exception as e:
            print(f"Failed to get model info: {e}")
            return {} 