"""
Training infrastructure for spiking neural networks.

Provides comprehensive training loops, optimization strategies, and monitoring
for event-based spiking neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
import time
import logging
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from .models import EventSNN, TrainingConfig
from .monitoring import get_metrics_collector
from .validation import validate_model_input, safe_operation, ValidationError
from .security import get_input_sanitizer


class SpikingTrainer:
    """Advanced trainer for spiking neural networks."""
    
    def __init__(
        self,
        model: EventSNN,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Logging and monitoring
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = get_metrics_collector()
        self.input_sanitizer = get_input_sanitizer()
        
        # Enhanced error tracking
        self.error_count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        elif self.config.lr_scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            return None
            
    @safe_operation
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch with enhanced error handling."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            try:
                # Validate inputs
                data = validate_model_input(data)
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Reset optimizer
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                
                # Compute loss
                loss = self.model.compute_loss(outputs, targets, self.config.loss_function)
                
                # Check for invalid loss
                if not torch.isfinite(loss):
                    self.logger.warning(f"Invalid loss detected at batch {batch_idx}, skipping")
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        raise RuntimeError("Too many consecutive training errors")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_value > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_value
                    )
                    # Log gradient norm if too large
                    if grad_norm > self.config.gradient_clip_value * 2:
                        self.logger.warning(f"Large gradient norm: {grad_norm:.2f}")
                        
                # Optimizer step
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                if targets.dim() == 1:  # Classification
                    predicted = outputs.argmax(dim=1)
                    total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                
                # Record metrics
                self.metrics_collector.record_events_processed(data.size(0))
                
                # Reset error counter on success
                self.consecutive_errors = 0
                
                # Update progress bar
                current_acc = total_correct / total_samples if total_samples > 0 else 0.0
                pbar.set_postfix({
                    'loss': f"{total_loss / (batch_idx + 1):.4f}",
                    'acc': f"{current_acc:.4f}"
                })
                
            except Exception as e:
                self.error_count += 1
                self.consecutive_errors += 1
                self.metrics_collector.record_error("training_batch")
                self.logger.error(f"Training batch {batch_idx} failed: {e}")
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    self.logger.error("Too many consecutive training errors, stopping epoch")
                    raise RuntimeError(f"Training failed after {self.consecutive_errors} consecutive errors")
                    
                continue
        
        epoch_time = time.time() - epoch_start_time
        self.logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
            'epoch_time': epoch_time,
            'error_count': self.error_count
        }
        
    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Validating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Compute loss
                loss = self.model.compute_loss(outputs, targets, self.config.loss_function)
                
                # Statistics
                total_loss += loss.item()
                if targets.dim() == 1:  # Classification
                    predicted = outputs.argmax(dim=1)
                    total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0
        }
        
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping and checkpointing."""
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Validation
            if val_loader:
                val_metrics = self.validate_epoch(val_loader, epoch)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                
                # Early stopping check
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    
                    # Save best model
                    if save_dir:
                        best_model_path = save_dir / "best_model.pth"
                        self.model.save_checkpoint(
                            str(best_model_path),
                            epoch,
                            val_metrics['loss']
                        )
                else:
                    self.patience_counter += 1
                    
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loader:
                        self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                    
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            # Logging
            epoch_time = time.time() - start_time
            log_msg = f"Epoch {epoch:3d} | Time: {epoch_time:.1f}s | "
            log_msg += f"Train Loss: {train_metrics['loss']:.4f} | "
            log_msg += f"Train Acc: {train_metrics['accuracy']:.4f}"
            
            if val_loader:
                log_msg += f" | Val Loss: {val_metrics['loss']:.4f} | "
                log_msg += f"Val Acc: {val_metrics['accuracy']:.4f}"
                
            log_msg += f" | LR: {current_lr:.6f}"
            self.logger.info(log_msg)
        
        # Save final model and history
        if save_dir:
            final_model_path = save_dir / "final_model.pth"
            self.model.save_checkpoint(
                str(final_model_path),
                epoch,
                history['train_loss'][-1]
            )
            
            # Save training history
            history_path = save_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
                
            # Save training plots
            self.plot_training_history(history, save_dir / "training_plots.png")
        
        return history
        
    def plot_training_history(
        self, 
        history: Dict[str, List[float]], 
        save_path: Optional[str] = None
    ):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy plot
        axes[0, 1].plot(history['train_accuracy'], label='Train Acc')
        if 'val_accuracy' in history and history['val_accuracy']:
            axes[0, 1].plot(history['val_accuracy'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Learning rate plot
        axes[1, 0].plot(history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        
        # Model statistics (placeholder)
        axes[1, 1].text(
            0.1, 0.5, 
            "Model Statistics\n" +
            f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n" +
            f"Best Val Loss: {self.best_val_loss:.4f}",
            transform=axes[1, 1].transAxes,
            fontsize=12,
            verticalalignment='center'
        )
        axes[1, 1].set_title('Model Info')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def evaluate(
        self,
        test_loader: DataLoader,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Evaluating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.model.compute_loss(outputs, targets, self.config.loss_function)
                
                total_loss += loss.item()
                
                if targets.dim() == 1:  # Classification
                    predicted = outputs.argmax(dim=1)
                    total_correct += (predicted == targets).sum().item()
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                total_samples += targets.size(0)
                
        results = {
            'loss': total_loss / len(test_loader),
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0
        }
        
        # Additional metrics
        if metrics and len(all_predictions) > 0:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            if 'precision' in metrics:
                results['precision'] = precision_score(
                    all_targets, all_predictions, average='weighted', zero_division=0
                )
            if 'recall' in metrics:
                results['recall'] = recall_score(
                    all_targets, all_predictions, average='weighted', zero_division=0
                )
            if 'f1' in metrics:
                results['f1'] = f1_score(
                    all_targets, all_predictions, average='weighted', zero_division=0
                )
                
        return results


class EventDataLoader:
    """Specialized data loader for event-based datasets."""
    
    @staticmethod
    def create_loaders(
        dataset_name: str,
        batch_size: int = 32,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test data loaders."""
        
        if dataset_name == "synthetic":
            from .core import EventDataset
            dataset = EventDataset.load("N-CARS")
            train_loader, val_loader = dataset.get_loaders(batch_size=batch_size)
            
            # Create test loader (same as val for now)
            test_loader = val_loader
            
            return train_loader, val_loader, test_loader
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
    @staticmethod
    def collate_events(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function for variable-length event sequences."""
        events_list, labels_list = zip(*batch)
        
        # Pad event sequences to same length
        max_events = max(events.shape[0] for events in events_list)
        
        padded_events = []
        for events in events_list:
            if events.shape[0] < max_events:
                padding = torch.zeros(max_events - events.shape[0], events.shape[1])
                events = torch.cat([events, padding], dim=0)
            padded_events.append(events)
            
        events_batch = torch.stack(padded_events)
        labels_batch = torch.stack(labels_list)
        
        return events_batch, labels_batch


def create_training_config(
    learning_rate: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 32,
    **kwargs
) -> TrainingConfig:
    """Create training configuration with sensible defaults."""
    config = TrainingConfig(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Update with any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            
    return config