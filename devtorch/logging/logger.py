import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter


class DevLogger:
    """
    Unified logging interface for DevTorch.
    
    Handles both console logging and TensorBoard logging in a simple, unified interface.
    
    Args:
        log_dir: Directory to save logs (default: ./logs)
        console_level: Console logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        use_tensorboard: Whether to enable TensorBoard logging
        experiment_name: Name for this experiment (used in log directory)
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        console_level: str = "INFO",
        use_tensorboard: bool = True,
        experiment_name: Optional[str] = None
    ):
        self.log_dir = log_dir
        self.console_level = console_level.upper()
        self.use_tensorboard = use_tensorboard
        
        # Create experiment-specific log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.run_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        else:
            self.run_dir = os.path.join(log_dir, f"run_{timestamp}")
        
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=self.run_dir)
        else:
            self.tb_writer = None
        
        # Logging state
        self.start_time = time.time()
        self._console_levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        self._current_level = self._console_levels[self.console_level]
        
        self.info(f"DevLogger initialized - Run directory: {self.run_dir}")
    
    def _should_log_console(self, level: str) -> bool:
        """Check if message should be logged to console based on level."""
        return self._console_levels.get(level.upper(), 1) >= self._current_level
    
    def _format_message(self, message: str, level: str = "INFO") -> str:
        """Format console message with timestamp and level."""
        elapsed = time.time() - self.start_time
        timestamp = time.strftime("%H:%M:%S")
        return f"[{timestamp}] [{level}] {message} (elapsed: {elapsed:.1f}s)"
    
    def debug(self, message: str):
        """Log debug message."""
        if self._should_log_console("DEBUG"):
            print(self._format_message(message, "DEBUG"))
    
    def info(self, message: str):
        """Log info message."""
        if self._should_log_console("INFO"):
            print(self._format_message(message, "INFO"))
    
    def warning(self, message: str):
        """Log warning message."""
        if self._should_log_console("WARNING"):
            print(self._format_message(message, "WARNING"))
    
    def error(self, message: str):
        """Log error message."""
        if self._should_log_console("ERROR"):
            print(self._format_message(message, "ERROR"))
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, prefix: str = ""):
        """
        Log metrics to both console and TensorBoard.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number for TensorBoard (optional)
            prefix: Prefix to add to metric names
        """
        # Log to console
        if self._should_log_console("INFO"):
            metric_strs = []
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metric_strs.append(f"{name}: {value:.4f}")
            
            console_msg = " | ".join(metric_strs)
            if prefix:
                console_msg = f"{prefix} - {console_msg}"
            
            self.info(console_msg)
        
        # Log to TensorBoard
        if self.tb_writer and step is not None:
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                full_name = f"{prefix}/{name}" if prefix else name
                self.tb_writer.add_scalar(full_name, value, step)
    
    def log_step(self, step: int, metrics: Dict[str, Union[float, int]], stage: str = "train"):
        """
        Log metrics for a training/validation step.
        
        Args:
            step: Step number
            metrics: Dictionary of metrics
            stage: Stage name ('train', 'val', etc.)
        """
        # Only log to TensorBoard for steps (too verbose for console)
        if self.tb_writer:
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                full_name = f"{stage}/{name}"
                self.tb_writer.add_scalar(full_name, value, step)
    
    def log_epoch(self, epoch: int, metrics: Dict[str, Union[float, int]]):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        # Console logging for epochs
        if self._should_log_console("INFO"):
            metric_strs = []
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metric_strs.append(f"{name}: {value:.4f}")
            
            console_msg = " | ".join(metric_strs)
            self.info(f"Epoch {epoch:03d} - {console_msg}")
        
        # TensorBoard logging
        if self.tb_writer:
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.tb_writer.add_scalar(name, value, epoch)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """
        Log histogram to TensorBoard.
        
        Args:
            name: Name of the histogram
            values: Tensor values to create histogram from
            step: Step number
        """
        if self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)
    
    def log_image(self, name: str, image: torch.Tensor, step: int):
        """
        Log image to TensorBoard.
        
        Args:
            name: Name of the image
            image: Image tensor (C, H, W) or (B, C, H, W)
            step: Step number
        """
        if self.tb_writer:
            self.tb_writer.add_image(name, image, step)
    
    def log_images(self, name: str, images: torch.Tensor, step: int):
        """
        Log multiple images to TensorBoard.
        
        Args:
            name: Name of the image grid
            images: Image tensor (B, C, H, W)
            step: Step number
        """
        if self.tb_writer:
            self.tb_writer.add_images(name, images, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor):
        """
        Log model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_to_model: Example input tensor
        """
        if self.tb_writer:
            try:
                self.tb_writer.add_graph(model, input_to_model)
                self.info("Model graph logged to TensorBoard")
            except Exception as e:
                self.warning(f"Failed to log model graph: {e}")
    
    def log_learning_rate(self, lr: float, step: int):
        """
        Log learning rate.
        
        Args:
            lr: Current learning rate
            step: Step number
        """
        if self.tb_writer:
            self.tb_writer.add_scalar("learning_rate", lr, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None):
        """
        Log hyperparameters to TensorBoard.
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Optional dictionary of final metrics
        """
        if self.tb_writer:
            try:
                self.tb_writer.add_hparams(hparams, metrics or {})
                self.info("Hyperparameters logged to TensorBoard")
            except Exception as e:
                self.warning(f"Failed to log hyperparameters: {e}")
    
    def log_text(self, name: str, text: str, step: int):
        """
        Log text to TensorBoard.
        
        Args:
            name: Name of the text log
            text: Text content
            step: Step number
        """
        if self.tb_writer:
            self.tb_writer.add_text(name, text, step)
    
    def save_text_log(self, filename: str, content: str):
        """
        Save text content to a file in the log directory.
        
        Args:
            filename: Name of the file
            content: Content to save
        """
        filepath = os.path.join(self.run_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        self.info(f"Saved text log: {filepath}")
    
    def log_training_start(self, model: torch.nn.Module, total_epochs: int, **kwargs):
        """
        Log training start information.
        
        Args:
            model: The model being trained
            total_epochs: Total number of epochs
            **kwargs: Additional training information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info("="*50)
        self.info("TRAINING START")
        self.info("="*50)
        self.info(f"Model: {model.__class__.__name__}")
        self.info(f"Total parameters: {total_params:,}")
        self.info(f"Trainable parameters: {trainable_params:,}")
        self.info(f"Total epochs: {total_epochs}")
        
        for key, value in kwargs.items():
            self.info(f"{key}: {value}")
        
        self.info("="*50)
    
    def log_training_end(self, total_time: float, best_metrics: Optional[Dict[str, float]] = None):
        """
        Log training end information.
        
        Args:
            total_time: Total training time in seconds
            best_metrics: Optional dictionary of best metrics achieved
        """
        self.info("="*50)
        self.info("TRAINING COMPLETE")
        self.info("="*50)
        self.info(f"Total training time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        
        if best_metrics:
            self.info("Best metrics achieved:")
            for name, value in best_metrics.items():
                self.info(f"  {name}: {value:.4f}")
        
        self.info(f"Logs saved to: {self.run_dir}")
        self.info("="*50)
    
    def flush(self):
        """Flush TensorBoard writer."""
        if self.tb_writer:
            self.tb_writer.flush()
    
    def close(self):
        """Close the logger and cleanup resources."""
        if self.tb_writer:
            self.tb_writer.close()
            self.info("TensorBoard writer closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ConsoleOnlyLogger:
    """
    Simplified logger that only does console logging.
    
    Useful for situations where you don't want TensorBoard overhead.
    """
    
    def __init__(self, level: str = "INFO"):
        self.level = level.upper()
        self._levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        self._current_level = self._levels[self.level]
        self.start_time = time.time()
    
    def _should_log(self, level: str) -> bool:
        return self._levels.get(level.upper(), 1) >= self._current_level
    
    def _format_message(self, message: str, level: str = "INFO") -> str:
        elapsed = time.time() - self.start_time
        timestamp = time.strftime("%H:%M:%S")
        return f"[{timestamp}] [{level}] {message} (elapsed: {elapsed:.1f}s)"
    
    def debug(self, message: str):
        if self._should_log("DEBUG"):
            print(self._format_message(message, "DEBUG"))
    
    def info(self, message: str):
        if self._should_log("INFO"):
            print(self._format_message(message, "INFO"))
    
    def warning(self, message: str):
        if self._should_log("WARNING"):
            print(self._format_message(message, "WARNING"))
    
    def error(self, message: str):
        if self._should_log("ERROR"):
            print(self._format_message(message, "ERROR"))
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, prefix: str = ""):
        if self._should_log("INFO"):
            metric_strs = []
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metric_strs.append(f"{name}: {value:.4f}")
            
            console_msg = " | ".join(metric_strs)
            if prefix:
                console_msg = f"{prefix} - {console_msg}"
            
            self.info(console_msg)
    
    def log_epoch(self, epoch: int, metrics: Dict[str, Union[float, int]]):
        if self._should_log("INFO"):
            metric_strs = []
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metric_strs.append(f"{name}: {value:.4f}")
            
            console_msg = " | ".join(metric_strs)
            self.info(f"Epoch {epoch:03d} - {console_msg}")
    
    def flush(self):
        pass
    
    def close(self):
        pass 