import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoints import CheckpointStrategy, load_checkpoint
from .strategies import ForwardStrategyManager
from ..logging import DevLogger


class ModelTrainer:
    """
    Main training engine for PyTorch models.
    
    Provides a clean interface for training with automatic checkpointing,
    logging, and validation without the complexity of PyTorch Lightning.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Optional validation data loader
        loss_fn: Loss function (or dict of loss functions for multi-head models)
        optimizer: Optimizer instance or config dict
        scheduler: Optional scheduler instance or config dict
        device: Device to train on ('cpu', 'cuda', or 'auto')
        save_dir: Directory to save checkpoints and logs
        checkpoint_strategies: List of checkpoint strategies to use
        logger: Optional logger instance
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Union[nn.Module, Dict[str, nn.Module], Callable] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "auto",
        save_dir: str = "./training_output",
        checkpoint_strategies: Optional[List[CheckpointStrategy]] = None,
        logger: Optional[DevLogger] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        forward_fn: Optional[Callable] = None
    ):
        self.model = model
        self.train_loader = train_loader  
        self.val_loader = val_loader
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = loss_fn
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.checkpoint_strategies = checkpoint_strategies or []
        self.logger = logger or DevLogger(save_dir)
        self.metrics = metrics or {}
        
        # Forward strategy management
        self.forward_manager = ForwardStrategyManager()
        if forward_fn is not None:
            self.forward_manager.register_custom_strategy(forward_fn)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        
        # For multi-head models
        self._is_multi_head = hasattr(model, 'decoder_names')
    
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of losses and metrics for this step
        """
        self.model.train()
        
        # Handle different batch formats
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch['inputs'], batch['targets']
        
        inputs = inputs.to(self.device)
        if isinstance(targets, dict):
            targets = {k: v.to(self.device) for k, v in targets.items()}
        else:
            targets = targets.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.forward_manager.forward(self.model, batch)
        
        loss_dict = self._calculate_loss(outputs, targets)
        total_loss = loss_dict.get('total', loss_dict.get('loss', list(loss_dict.values())[0]))
        
        total_loss.backward()
        self.optimizer.step()
        
        metrics_dict = self._calculate_metrics(outputs, targets, 'train')
        
        # Combine losses and metrics
        step_results = {**loss_dict, **metrics_dict}
        
        return step_results
    
    def val_step(self, batch: Any) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            batch: Batch of validation data
            
        Returns:
            Dictionary of losses and metrics for this step
        """
        self.model.eval()
        
        with torch.no_grad():
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch['inputs'], batch['targets']
            
            inputs = inputs.to(self.device)
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                targets = targets.to(self.device)
            
            outputs = self.forward_manager.forward(self.model, batch)
            
            loss_dict = self._calculate_loss(outputs, targets)
            metrics_dict = self._calculate_metrics(outputs, targets, 'val')
            
            # Combine losses and metrics
            step_results = {**loss_dict, **metrics_dict}
            
            return step_results
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {}
        num_batches = len(self.train_loader)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            step_results = self.train_step(batch)
            self.global_step += 1
            
            # Accumulate metrics
            for key, value in step_results.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value.item() if torch.is_tensor(value) else value)
            
            # Log step-level metrics periodically
            if batch_idx % 50 == 0:
                self.logger.log_step(self.global_step, step_results, 'train')
        
        # Average metrics over epoch
        epoch_averages = {}
        for key, values in epoch_metrics.items():
            epoch_averages[f'train/{key}'] = sum(values) / len(values)
        
        epoch_time = time.time() - start_time
        epoch_averages['train/epoch_time'] = epoch_time
        
        return epoch_averages
    
    def val_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {}
        
        epoch_metrics = {}
        
        for batch in self.val_loader:
            step_results = self.val_step(batch)
            
            # Accumulate metrics
            for key, value in step_results.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value.item() if torch.is_tensor(value) else value)
        
        # Average metrics over epoch
        epoch_averages = {}
        for key, values in epoch_metrics.items():
            epoch_averages[f'val/{key}'] = sum(values) / len(values)
        
        return epoch_averages
    
    def train(
        self, 
        epochs: int,
        resume_from: Optional[str] = None,
        validate_every: int = 1,
        log_every: int = 1
    ):
        """
        Main training loop.
        
        Args:
            epochs: Number of epochs to train
            resume_from: Optional checkpoint path to resume from
            validate_every: Run validation every N epochs
            log_every: Log metrics every N epochs
        """
        # Resume from checkpoint if specified
        if resume_from:
            checkpoint_info = load_checkpoint(resume_from, self.model, self.optimizer, self.scheduler, self.device)
            self.current_epoch = checkpoint_info['epoch'] + 1
            self.logger.info(f"Resumed training from epoch {self.current_epoch}")
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self.train_epoch()
            
            # Validation epoch
            val_metrics = {}
            if epoch % validate_every == 0:
                val_metrics = self.val_epoch()
            
            # Combine all metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs a metric value
                    metric_key = 'val/loss' if 'val/loss' in all_metrics else 'train/loss'
                    if metric_key in all_metrics:
                        self.scheduler.step(all_metrics[metric_key])
                else:
                    self.scheduler.step()
            
            # Log epoch metrics
            if epoch % log_every == 0:
                self.logger.log_epoch(epoch, all_metrics)
            
            # Save checkpoints
            self._save_checkpoints(epoch, all_metrics)
            
            # Check for early stopping
            if self._should_early_stop():
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        self.logger.info("Training completed!")
        self.logger.close()
    
    def _calculate_loss(self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                       targets: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Calculate loss for single or multi-head models."""
        if isinstance(self.loss_fn, dict) and isinstance(outputs, dict):
            # Multi-head model with multiple loss functions
            losses = {}
            total_loss = 0
            
            for head_name, head_output in outputs.items():
                if head_name in self.loss_fn and head_name in targets:
                    loss = self.loss_fn[head_name](head_output, targets[head_name])
                    losses[f'{head_name}_loss'] = loss
                    total_loss += loss
            
            losses['total'] = total_loss
            return losses
        
        elif isinstance(outputs, dict):
            # Multi-head model with single loss function
            losses = {}
            total_loss = 0
            
            for head_name, head_output in outputs.items():
                if head_name in targets:
                    loss = self.loss_fn(head_output, targets[head_name])
                    losses[f'{head_name}_loss'] = loss
                    total_loss += loss
            
            losses['total'] = total_loss
            return losses
        
        else:
            # Single-head model
            loss = self.loss_fn(outputs, targets)
            return {'loss': loss}
    
    def _calculate_metrics(self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                          targets: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                          stage: str) -> Dict[str, float]:
        """Calculate metrics for single or multi-head models."""
        metrics = {}
        
        if isinstance(outputs, dict):
            # Multi-head model
            for head_name, head_output in outputs.items():
                if head_name in targets:
                    head_targets = targets[head_name]
                else:
                    head_targets = targets
                
                for metric_name, metric_fn in self.metrics.items():
                    try:
                        metric_value = metric_fn(head_output, head_targets)
                        metrics[f'{head_name}_{metric_name}'] = metric_value
                    except Exception:
                        # Skip metrics that don't apply to this head
                        pass
        else:
            # Single-head model
            for metric_name, metric_fn in self.metrics.items():
                try:
                    metric_value = metric_fn(outputs, targets)
                    metrics[metric_name] = metric_value
                except Exception:
                    pass
        
        return metrics
    
    def _save_checkpoints(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoints using all configured strategies."""
        for strategy in self.checkpoint_strategies:
            filepath = strategy.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=metrics,
                scheduler=self.scheduler,
                additional_info={'global_step': self.global_step}
            )
            
            if filepath:
                self.logger.info(f"Saved checkpoint: {filepath}")
    
    def _should_early_stop(self) -> bool:
        """Check if any checkpoint strategy indicates early stopping."""
        for strategy in self.checkpoint_strategies:
            if hasattr(strategy, 'is_early_stopping') and strategy.is_early_stopping():
                return True
        return False
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = f"""
Model Summary:
==============
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Model device: {next(self.model.parameters()).device}
"""
        
        if self._is_multi_head:
            summary += f"Multi-head model with heads: {self.model.decoder_names}\n"
        
        return summary 