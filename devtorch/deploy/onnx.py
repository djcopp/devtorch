import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class ONNXExporter:
    """
    Export PyTorch models to ONNX format with validation and usage code generation.
    
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
    
    def export(
        self,
        example_input: torch.Tensor,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[dict] = None,
        opset_version: int = 11,
        validate: bool = True,
        generate_usage_code: bool = True
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            example_input: Example input tensor for tracing
            input_names: Names for input tensors
            output_names: Names for output tensors  
            dynamic_axes: Dynamic axes specification for variable input sizes
            opset_version: ONNX opset version to use
            validate: Whether to validate the exported model
            generate_usage_code: Whether to generate usage code
            
        Returns:
            Path to the exported ONNX model
        """
        onnx_path = self.export_dir / f"{self.model_name}.onnx"
        
        print(f"Exporting model to ONNX: {onnx_path}")
        
        # Default input/output names
        if input_names is None:
            if isinstance(example_input, (list, tuple)):
                input_names = [f"input_{i}" for i in range(len(example_input))]
            else:
                input_names = ["input"]
        
        if output_names is None:
            # Try to infer from model output
            with torch.no_grad():
                sample_output = self.model(example_input)
            
            if isinstance(sample_output, dict):
                output_names = list(sample_output.keys())
            elif isinstance(sample_output, (list, tuple)):
                output_names = [f"output_{i}" for i in range(len(sample_output))]
            else:
                output_names = ["output"]
        
        # Export to ONNX
        try:
            torch.onnx.export(
                self.model,
                example_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            print(f"✓ Successfully exported to {onnx_path}")
            
        except Exception as e:
            print(f"✗ Export failed: {e}")
            raise
        
        # Validate exported model
        if validate:
            self._validate_onnx_model(str(onnx_path), example_input)
        
        # Generate usage code
        if generate_usage_code:
            self._generate_usage_code(
                onnx_path, 
                example_input, 
                input_names, 
                output_names
            )
        
        return str(onnx_path)
    
    def _validate_onnx_model(self, onnx_path: str, example_input: torch.Tensor):
        """Validate that the ONNX model produces the same outputs as PyTorch."""
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            print("⚠ ONNX validation requires 'onnx' and 'onnxruntime' packages")
            return
        
        print("Validating ONNX model...")
        
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get PyTorch output
            with torch.no_grad():
                pytorch_output = self.model(example_input)
            
            # Prepare input for ONNX Runtime
            if isinstance(example_input, (list, tuple)):
                ort_inputs = {
                    input_meta.name: inp.numpy() 
                    for input_meta, inp in zip(ort_session.get_inputs(), example_input)
                }
            else:
                ort_inputs = {ort_session.get_inputs()[0].name: example_input.numpy()}
            
            # Get ONNX output
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Compare outputs
            if isinstance(pytorch_output, dict):
                pytorch_outputs = list(pytorch_output.values())
            elif isinstance(pytorch_output, (list, tuple)):
                pytorch_outputs = list(pytorch_output)
            else:
                pytorch_outputs = [pytorch_output]
            
            # Check if outputs match
            all_close = True
            for i, (pt_out, onnx_out) in enumerate(zip(pytorch_outputs, ort_outputs)):
                if not torch.allclose(pt_out, torch.from_numpy(onnx_out), rtol=1e-3, atol=1e-5):
                    print(f"⚠ Output {i} differs between PyTorch and ONNX")
                    all_close = False
            
            if all_close:
                print("✓ ONNX model validation passed")
            else:
                print("⚠ ONNX model validation failed - outputs differ")
                
        except Exception as e:
            print(f"⚠ ONNX validation failed: {e}")
    
    def _generate_usage_code(
        self,
        onnx_path: Path,
        example_input: torch.Tensor,
        input_names: List[str],
        output_names: List[str]
    ):
        """Generate Python code showing how to use the exported ONNX model."""
        
        # Get input shape information
        if isinstance(example_input, (list, tuple)):
            input_shapes = [list(inp.shape) for inp in example_input]
        else:
            input_shapes = [list(example_input.shape)]
        
        usage_code = f'''"""
ONNX Model Usage Code for {self.model_name}
Generated by DevTorch

This code shows how to load and use the exported ONNX model.
"""

import numpy as np
import onnxruntime as ort

def load_model(model_path: str = "{onnx_path.name}"):
    """Load the ONNX model."""
    session = ort.InferenceSession(model_path)
    return session

def predict(session, input_data):
    """
    Run inference on the ONNX model.
    
    Args:
        session: ONNX Runtime session
        input_data: Input data (numpy array or list of arrays)
    
    Returns:
        Model outputs
    """
    # Prepare inputs
'''

        if len(input_names) == 1:
            usage_code += f'''    inputs = {{"{input_names[0]}": input_data}}
    
    # Expected input shape: {input_shapes[0]}
    # Make sure your input data matches this shape
    assert input_data.shape == {tuple(input_shapes[0])}, f"Expected shape {tuple(input_shapes[0])}, got {{input_data.shape}}"
'''
        else:
            usage_code += '''    if isinstance(input_data, (list, tuple)):
        inputs = {
'''
            for i, (name, shape) in enumerate(zip(input_names, input_shapes)):
                usage_code += f'''            "{name}": input_data[{i}],  # Expected shape: {shape}
'''
            usage_code += '''        }
    else:
        raise ValueError("Multi-input model requires list or tuple of inputs")
'''

        usage_code += f'''
    # Run inference
    outputs = session.run([{", ".join(f'"{name}"' for name in output_names)}], inputs)
    
    return outputs

def example_usage():
    """Example of how to use the model."""
    # Load model
    session = load_model()
    
    # Create example input
'''

        if len(input_shapes) == 1:
            usage_code += f'''    example_input = np.random.randn{tuple(input_shapes[0])}.astype(np.float32)
    
    # Run prediction
    outputs = predict(session, example_input)
    
    print(f"Input shape: {{example_input.shape}}")
'''
        else:
            usage_code += '''    example_inputs = [
'''
            for i, shape in enumerate(input_shapes):
                usage_code += f'''        np.random.randn{tuple(shape)}.astype(np.float32),  # {input_names[i]}
'''
            usage_code += '''    ]
    
    # Run prediction
    outputs = predict(session, example_inputs)
    
    print("Input shapes:", [inp.shape for inp in example_inputs])
'''

        for i, name in enumerate(output_names):
            usage_code += f'''    print(f"{name} shape: {{outputs[{i}].shape}}")
'''

        usage_code += '''

if __name__ == "__main__":
    example_usage()
'''

        # Save usage code
        usage_file = self.export_dir / f"{self.model_name}_usage.py"
        with open(usage_file, 'w') as f:
            f.write(usage_code)
        
        print(f"✓ Generated usage code: {usage_file}")
        
        # Also create a requirements file
        requirements = """# Requirements for using the exported ONNX model
onnxruntime>=1.8.0
numpy>=1.19.0
"""
        
        req_file = self.export_dir / "requirements.txt"
        with open(req_file, 'w') as f:
            f.write(requirements)
        
        print(f"✓ Generated requirements file: {req_file}")
    
    def export_with_optimization(
        self,
        example_input: torch.Tensor,
        optimization_level: str = "basic",
        **kwargs
    ) -> str:
        """
        Export model with optimization.
        
        Args:
            example_input: Example input for tracing
            optimization_level: Optimization level ('basic', 'extended', 'all')
            **kwargs: Additional arguments for export
            
        Returns:
            Path to optimized ONNX model
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("⚠ Optimization requires 'onnxruntime' package")
            return self.export(example_input, **kwargs)
        
        # First export normally
        onnx_path = self.export(example_input, validate=False, **kwargs)
        
        # Optimize the model
        print("Optimizing ONNX model...")
        
        optimized_path = self.export_dir / f"{self.model_name}_optimized.onnx"
        
        # Set optimization level
        if optimization_level == "basic":
            graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif optimization_level == "extended":
            graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        elif optimization_level == "all":
            graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            raise ValueError(f"Unknown optimization level: {optimization_level}")
        
        # Create session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = graph_optimization_level
        sess_options.optimized_model_filepath = str(optimized_path)
        
        # Create session to trigger optimization
        _ = ort.InferenceSession(onnx_path, sess_options)
        
        print(f"✓ Optimized model saved: {optimized_path}")
        
        # Validate optimized model
        self._validate_onnx_model(str(optimized_path), example_input)
        
        return str(optimized_path)
    
    def get_model_info(self, onnx_path: str) -> dict:
        """Get information about the exported ONNX model."""
        try:
            import onnx
        except ImportError:
            print("Model info requires 'onnx' package")
            return {}
        
        model = onnx.load(onnx_path)
        
        info = {
            "ir_version": model.ir_version,
            "opset_version": model.opset_import[0].version if model.opset_import else None,
            "producer_name": model.producer_name,
            "model_version": model.model_version,
            "inputs": [],
            "outputs": []
        }
        
        # Get input info
        for input_info in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" 
                    for dim in input_info.type.tensor_type.shape.dim]
            info["inputs"].append({
                "name": input_info.name,
                "shape": shape,
                "type": input_info.type.tensor_type.elem_type
            })
        
        # Get output info
        for output_info in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" 
                    for dim in output_info.type.tensor_type.shape.dim]
            info["outputs"].append({
                "name": output_info.name,
                "shape": shape,
                "type": output_info.type.tensor_type.elem_type
            })
        
        return info 