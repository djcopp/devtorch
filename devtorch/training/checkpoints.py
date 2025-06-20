import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch
import torch.nn as nn


class CheckpointStrategy(ABC):
    """Base class for checkpoint strategies."""
    
    def __init__(self, save_dir: str, filename_prefix: str = "model"):
        self.save_dir = save_dir
        self.filename_prefix = filename_prefix
        os.makedirs(save_dir, exist_ok=True)
    
    @abstractmethod
    def should_save(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Determine if a checkpoint should be saved."""
        pass
    
    @abstractmethod
    def get_filename(self, epoch: int, metrics: Dict[str, float]) -> str:
        """Generate filename for the checkpoint."""
        pass
    
    def save_checkpoint(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        epoch: int, 
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        additional_info: Optional[Dict] = None
    ) -> Optional[str]:
        """Save checkpoint if strategy conditions are met."""
        if not self.should_save(epoch, metrics):
            return None
        
        filename = self.get_filename(epoch, metrics)
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        self._post_save_hook(filepath, epoch, metrics)
        
        return filepath
    
    def _post_save_hook(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Hook called after saving checkpoint. Can be overridden by subclasses."""
        pass


class LatestModelCheckpoint(CheckpointStrategy):
    """Save the latest model checkpoint, optionally keeping only the last N checkpoints."""
    
    def __init__(
        self, 
        save_dir: str, 
        filename_prefix: str = "latest",
        save_every_n_epochs: int = 1,
        keep_last_n: int = 3
    ):
        super().__init__(save_dir, filename_prefix)
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_last_n = keep_last_n
        self.saved_checkpoints = []
    
    def should_save(self, epoch: int, metrics: Dict[str, float]) -> bool:
        return epoch % self.save_every_n_epochs == 0
    
    def get_filename(self, epoch: int, metrics: Dict[str, float]) -> str:
        return f"{self.filename_prefix}_epoch_{epoch:03d}.pt"
    
    def _post_save_hook(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        self.saved_checkpoints.append(filepath)
        
        # Remove old checkpoints if we exceed the limit
        while len(self.saved_checkpoints) > self.keep_last_n:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)


class BestModelCheckpoint(CheckpointStrategy):
    """Save the best model based on a specific metric."""
    
    def __init__(
        self, 
        save_dir: str,
        monitor_metric: str,
        mode: str = "min",
        filename_prefix: Optional[str] = None,
        min_delta: float = 0.0,
        patience: Optional[int] = None
    ):
        if filename_prefix is None:
            filename_prefix = f"best_{monitor_metric.replace('/', '_')}"
        
        super().__init__(save_dir, filename_prefix)
        
        self.monitor_metric = monitor_metric
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.patience = patience
        
        if self.mode not in ["min", "max"]:
            raise ValueError("Mode must be 'min' or 'max'")
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_epoch = -1
        self.patience_counter = 0
        self.best_filepath = None
    
    def should_save(self, epoch: int, metrics: Dict[str, float]) -> bool:
        if self.monitor_metric not in metrics:
            return False
        
        current_value = metrics[self.monitor_metric]
        
        if self.mode == "min":
            is_better = current_value < (self.best_value - self.min_delta)
        else:
            is_better = current_value > (self.best_value + self.min_delta)
        
        if is_better:
            self.best_value = current_value
            self.best_epoch = epoch
            self.patience_counter = 0
            return True
        else:
            if self.patience is not None:
                self.patience_counter += 1
            return False
    
    def get_filename(self, epoch: int, metrics: Dict[str, float]) -> str:
        metric_value = metrics[self.monitor_metric]
        return f"{self.filename_prefix}_epoch_{epoch:03d}_metric_{metric_value:.4f}.pt"
    
    def _post_save_hook(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        # Remove previous best checkpoint
        if self.best_filepath and os.path.exists(self.best_filepath):
            os.remove(self.best_filepath)
        
        self.best_filepath = filepath
    
    def is_early_stopping(self) -> bool:
        """Check if early stopping criteria is met."""
        return self.patience is not None and self.patience_counter >= self.patience


class MultiMetricCheckpoint(CheckpointStrategy):
    """Save checkpoints based on multiple metrics."""
    
    def __init__(
        self, 
        save_dir: str,
        metric_configs: Dict[str, Dict[str, Union[str, float]]],
        filename_prefix: str = "multi_best"
    ):
        super().__init__(save_dir, filename_prefix)
        
        self.metric_configs = metric_configs
        self.best_values = {}
        self.best_epochs = {}
        self.best_filepaths = {}
        
        for metric_name, config in metric_configs.items():
            mode = config.get('mode', 'min').lower()
            if mode not in ['min', 'max']:
                raise ValueError(f"Mode for {metric_name} must be 'min' or 'max'")
            
            self.best_values[metric_name] = float('inf') if mode == 'min' else float('-inf')
            self.best_epochs[metric_name] = -1
            self.best_filepaths[metric_name] = None
    
    def should_save(self, epoch: int, metrics: Dict[str, float]) -> bool:
        should_save_any = False
        
        for metric_name, config in self.metric_configs.items():
            if metric_name not in metrics:
                continue
            
            current_value = metrics[metric_name]
            mode = config.get('mode', 'min').lower()
            min_delta = config.get('min_delta', 0.0)
            
            if mode == 'min':
                is_better = current_value < (self.best_values[metric_name] - min_delta)
            else:
                is_better = current_value > (self.best_values[metric_name] + min_delta)
            
            if is_better:
                self.best_values[metric_name] = current_value
                self.best_epochs[metric_name] = epoch
                should_save_any = True
        
        return should_save_any
    
    def get_filename(self, epoch: int, metrics: Dict[str, float]) -> str:
        return f"{self.filename_prefix}_epoch_{epoch:03d}.pt"
    
    def _post_save_hook(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        # This saves one file for all improved metrics
        # Alternative implementation could save separate files per metric
        pass


def load_checkpoint(
    filepath: str, 
    model: nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Dict:
    """
    Load a checkpoint and restore model/optimizer/scheduler states.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into  
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint information (epoch, metrics, etc.)
    """
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'additional_info': {k: v for k, v in checkpoint.items() 
                           if k not in ['epoch', 'model_state_dict', 'optimizer_state_dict', 
                                       'scheduler_state_dict', 'metrics']}
    } 