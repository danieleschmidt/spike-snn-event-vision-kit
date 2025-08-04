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
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
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
            
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Reset optimizer
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Compute loss
            loss = self.model.compute_loss(outputs, targets, self.config.loss_function)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_value
                )
                
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            if targets.dim() == 1:  # Classification
                predicted = outputs.argmax(dim=1)
                total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            # Update progress bar
            current_acc = total_correct / total_samples if total_samples > 0 else 0.0
            pbar.set_postfix({
                'loss': f"{total_loss / (batch_idx + 1):.4f}",
                'acc': f"{current_acc:.4f}"
            })
            
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0
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
            history['train_loss'].append(train_metrics['loss'])\n            history['train_accuracy'].append(train_metrics['accuracy'])\n            \n            # Validation\n            if val_loader:\n                val_metrics = self.validate_epoch(val_loader, epoch)\n                history['val_loss'].append(val_metrics['loss'])\n                history['val_accuracy'].append(val_metrics['accuracy'])\n                \n                # Early stopping check\n                if val_metrics['loss'] < self.best_val_loss:\n                    self.best_val_loss = val_metrics['loss']\n                    self.patience_counter = 0\n                    \n                    # Save best model\n                    if save_dir:\n                        best_model_path = save_dir / \"best_model.pth\"\n                        self.model.save_checkpoint(\n                            str(best_model_path),\n                            epoch,\n                            val_metrics['loss']\n                        )\n                else:\n                    self.patience_counter += 1\n                    \n                # Early stopping\n                if self.patience_counter >= self.config.early_stopping_patience:\n                    self.logger.info(f\"Early stopping at epoch {epoch}\")\n                    break\n                    \n            # Learning rate scheduling\n            if self.scheduler:\n                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):\n                    if val_loader:\n                        self.scheduler.step(val_metrics['loss'])\n                else:\n                    self.scheduler.step()\n                    \n            # Record learning rate\n            current_lr = self.optimizer.param_groups[0]['lr']\n            history['learning_rate'].append(current_lr)\n            \n            # Logging\n            epoch_time = time.time() - start_time\n            log_msg = f\"Epoch {epoch:3d} | Time: {epoch_time:.1f}s | \"\n            log_msg += f\"Train Loss: {train_metrics['loss']:.4f} | \"\n            log_msg += f\"Train Acc: {train_metrics['accuracy']:.4f}\"\n            \n            if val_loader:\n                log_msg += f\" | Val Loss: {val_metrics['loss']:.4f} | \"\n                log_msg += f\"Val Acc: {val_metrics['accuracy']:.4f}\"\n                \n            log_msg += f\" | LR: {current_lr:.6f}\"\n            self.logger.info(log_msg)\n        \n        # Save final model and history\n        if save_dir:\n            final_model_path = save_dir / \"final_model.pth\"\n            self.model.save_checkpoint(\n                str(final_model_path),\n                epoch,\n                history['train_loss'][-1]\n            )\n            \n            # Save training history\n            history_path = save_dir / \"training_history.json\"\n            with open(history_path, 'w') as f:\n                json.dump(history, f, indent=2)\n                \n            # Save training plots\n            self.plot_training_history(history, save_dir / \"training_plots.png\")\n        \n        return history\n        \n    def plot_training_history(\n        self, \n        history: Dict[str, List[float]], \n        save_path: Optional[str] = None\n    ):\n        \"\"\"Plot training history.\"\"\"\n        fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n        \n        # Loss plot\n        axes[0, 0].plot(history['train_loss'], label='Train Loss')\n        if 'val_loss' in history and history['val_loss']:\n            axes[0, 0].plot(history['val_loss'], label='Val Loss')\n        axes[0, 0].set_title('Loss')\n        axes[0, 0].set_xlabel('Epoch')\n        axes[0, 0].set_ylabel('Loss')\n        axes[0, 0].legend()\n        \n        # Accuracy plot\n        axes[0, 1].plot(history['train_accuracy'], label='Train Acc')\n        if 'val_accuracy' in history and history['val_accuracy']:\n            axes[0, 1].plot(history['val_accuracy'], label='Val Acc')\n        axes[0, 1].set_title('Accuracy')\n        axes[0, 1].set_xlabel('Epoch')\n        axes[0, 1].set_ylabel('Accuracy')\n        axes[0, 1].legend()\n        \n        # Learning rate plot\n        axes[1, 0].plot(history['learning_rate'])\n        axes[1, 0].set_title('Learning Rate')\n        axes[1, 0].set_xlabel('Epoch')\n        axes[1, 0].set_ylabel('Learning Rate')\n        axes[1, 0].set_yscale('log')\n        \n        # Model statistics (placeholder)\n        axes[1, 1].text(\n            0.1, 0.5, \n            \"Model Statistics\\n\" +\n            f\"Parameters: {sum(p.numel() for p in self.model.parameters()):,}\\n\" +\n            f\"Best Val Loss: {self.best_val_loss:.4f}\",\n            transform=axes[1, 1].transAxes,\n            fontsize=12,\n            verticalalignment='center'\n        )\n        axes[1, 1].set_title('Model Info')\n        axes[1, 1].axis('off')\n        \n        plt.tight_layout()\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n        plt.show()\n        \n    def evaluate(\n        self,\n        test_loader: DataLoader,\n        metrics: Optional[List[str]] = None\n    ) -> Dict[str, float]:\n        \"\"\"Evaluate model on test set.\"\"\"\n        self.model.eval()\n        \n        total_loss = 0.0\n        total_correct = 0\n        total_samples = 0\n        all_predictions = []\n        all_targets = []\n        \n        with torch.no_grad():\n            for data, targets in tqdm(test_loader, desc=\"Evaluating\"):\n                data, targets = data.to(self.device), targets.to(self.device)\n                \n                outputs = self.model(data)\n                loss = self.model.compute_loss(outputs, targets, self.config.loss_function)\n                \n                total_loss += loss.item()\n                \n                if targets.dim() == 1:  # Classification\n                    predicted = outputs.argmax(dim=1)\n                    total_correct += (predicted == targets).sum().item()\n                    all_predictions.extend(predicted.cpu().numpy())\n                    all_targets.extend(targets.cpu().numpy())\n                    \n                total_samples += targets.size(0)\n                \n        results = {\n            'loss': total_loss / len(test_loader),\n            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0\n        }\n        \n        # Additional metrics\n        if metrics and len(all_predictions) > 0:\n            from sklearn.metrics import precision_score, recall_score, f1_score\n            \n            if 'precision' in metrics:\n                results['precision'] = precision_score(\n                    all_targets, all_predictions, average='weighted', zero_division=0\n                )\n            if 'recall' in metrics:\n                results['recall'] = recall_score(\n                    all_targets, all_predictions, average='weighted', zero_division=0\n                )\n            if 'f1' in metrics:\n                results['f1'] = f1_score(\n                    all_targets, all_predictions, average='weighted', zero_division=0\n                )\n                \n        return results


class EventDataLoader:\n    \"\"\"Specialized data loader for event-based datasets.\"\"\"\n    \n    @staticmethod\n    def create_loaders(\n        dataset_name: str,\n        batch_size: int = 32,\n        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),\n        **kwargs\n    ) -> Tuple[DataLoader, DataLoader, DataLoader]:\n        \"\"\"Create train/val/test data loaders.\"\"\"\n        \n        if dataset_name == \"synthetic\":\n            from .core import EventDataset\n            dataset = EventDataset.load(\"N-CARS\")\n            train_loader, val_loader = dataset.get_loaders(batch_size=batch_size)\n            \n            # Create test loader (same as val for now)\n            test_loader = val_loader\n            \n            return train_loader, val_loader, test_loader\n        else:\n            raise ValueError(f\"Unknown dataset: {dataset_name}\")\n            \n    @staticmethod\n    def collate_events(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:\n        \"\"\"Custom collate function for variable-length event sequences.\"\"\"\n        events_list, labels_list = zip(*batch)\n        \n        # Pad event sequences to same length\n        max_events = max(events.shape[0] for events in events_list)\n        \n        padded_events = []\n        for events in events_list:\n            if events.shape[0] < max_events:\n                padding = torch.zeros(max_events - events.shape[0], events.shape[1])\n                events = torch.cat([events, padding], dim=0)\n            padded_events.append(events)\n            \n        events_batch = torch.stack(padded_events)\n        labels_batch = torch.stack(labels_list)\n        \n        return events_batch, labels_batch


def create_training_config(\n    learning_rate: float = 1e-3,\n    epochs: int = 100,\n    batch_size: int = 32,\n    **kwargs\n) -> TrainingConfig:\n    \"\"\"Create training configuration with sensible defaults.\"\"\"\n    config = TrainingConfig(\n        learning_rate=learning_rate,\n        epochs=epochs,\n        batch_size=batch_size\n    )\n    \n    # Update with any additional kwargs\n    for key, value in kwargs.items():\n        if hasattr(config, key):\n            setattr(config, key, value)\n            \n    return config\n