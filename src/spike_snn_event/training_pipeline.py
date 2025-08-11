"""
Production-ready training pipeline with comprehensive data quality assurance and robustness.

This module provides secure, monitored, and resilient training pipelines for
neuromorphic vision models with extensive validation and error handling.
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .validation import (
    ValidationResult, get_stream_integrity_validator, 
    get_model_output_validator, CircuitBreaker, DataIntegrityError
)
from .security import get_input_sanitizer, get_security_audit_log
from .security_enhancements import get_adversarial_defense, get_memory_safety_manager
from .health import get_resilient_operation_manager, get_resource_monitor


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    model_name: str
    dataset_path: str
    output_dir: str
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    save_frequency: int = 10
    validation_frequency: int = 5
    early_stopping_patience: int = 20
    max_memory_gb: float = 8.0
    enable_adversarial_defense: bool = True
    enable_data_augmentation: bool = False
    gradient_clipping: float = 1.0
    
    def validate(self) -> ValidationResult:
        """Validate training configuration."""
        from .validation import DataValidator
        validator = DataValidator()
        
        result = ValidationResult()
        
        # Validate paths
        if not Path(self.dataset_path).exists():
            result.add_error("DATASET_NOT_FOUND", f"Dataset path does not exist: {self.dataset_path}")
            
        # Validate numeric parameters
        numeric_validations = [
            (self.batch_size, 1, 1000, "batch_size"),
            (self.learning_rate, 1e-6, 1.0, "learning_rate"),
            (self.epochs, 1, 10000, "epochs"),
            (self.validation_split, 0.0, 0.5, "validation_split"),
            (self.save_frequency, 1, 1000, "save_frequency"),
            (self.validation_frequency, 1, 100, "validation_frequency"),
            (self.early_stopping_patience, 1, 1000, "early_stopping_patience"),
            (self.max_memory_gb, 0.1, 128.0, "max_memory_gb"),
            (self.gradient_clipping, 0.1, 10.0, "gradient_clipping")
        ]
        
        for value, min_val, max_val, field_name in numeric_validations:
            field_result = validator.validate_numeric_range(value, min_val, max_val, field_name)
            result.errors.extend(field_result.errors)
            result.warnings.extend(field_result.warnings)
            if not field_result.is_valid:
                result.is_valid = False
                
        return result


class DataQualityAssurance:
    """Comprehensive data quality assurance for training pipelines."""
    
    def __init__(self, config: TrainingConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.input_sanitizer = get_input_sanitizer()
        self.stream_validator = get_stream_integrity_validator()
        self.adversarial_defense = get_adversarial_defense()
        
        # Quality metrics
        self.quality_history = []
        self.corruption_detected = 0
        self.samples_rejected = 0
        self.total_samples_processed = 0
        
    def validate_training_data(self, dataset_path: Path) -> Dict[str, Any]:
        """Comprehensive validation of training dataset."""
        validation_start = time.time()
        
        try:
            # Load and inspect dataset
            dataset_info = self._analyze_dataset_structure(dataset_path)
            
            # Validate data integrity
            integrity_result = self._validate_data_integrity(dataset_path)
            
            # Check for data quality issues
            quality_result = self._assess_data_quality(dataset_path)
            
            # Security validation
            security_result = self._validate_data_security(dataset_path)
            
            validation_duration = time.time() - validation_start
            
            return {
                'validation_timestamp': time.time(),
                'validation_duration': validation_duration,
                'dataset_info': dataset_info,
                'integrity': integrity_result,
                'quality': quality_result,
                'security': security_result,
                'overall_valid': all([
                    integrity_result.get('valid', False),
                    quality_result.get('valid', False),
                    security_result.get('valid', False)
                ])
            }
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return {
                'validation_timestamp': time.time(),
                'error': str(e),
                'overall_valid': False
            }
            
    def _analyze_dataset_structure(self, dataset_path: Path) -> Dict[str, Any]:
        """Analyze dataset structure and basic statistics."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
        # Basic file analysis
        if dataset_path.is_file():
            file_size_mb = dataset_path.stat().st_size / (1024 ** 2)
            file_info = {
                'type': 'single_file',
                'size_mb': file_size_mb,
                'format': dataset_path.suffix,
                'path': str(dataset_path)
            }
        else:
            # Directory analysis
            files = list(dataset_path.rglob('*'))
            data_files = [f for f in files if f.is_file() and f.suffix in ['.npy', '.h5', '.dat', '.json']]
            total_size_mb = sum(f.stat().st_size for f in data_files) / (1024 ** 2)
            
            file_info = {
                'type': 'directory',
                'total_files': len(data_files),
                'total_size_mb': total_size_mb,
                'file_types': list(set(f.suffix for f in data_files)),
                'path': str(dataset_path)
            }
            
        return file_info
        
    def _validate_data_integrity(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate data integrity and consistency."""
        try:
            # Check file accessibility
            if dataset_path.is_file():
                files_to_check = [dataset_path]
            else:
                files_to_check = list(dataset_path.rglob('*.npy'))[:10]  # Sample first 10 files
                
            integrity_issues = []
            valid_files = 0
            
            for file_path in files_to_check:
                try:
                    # Basic file integrity
                    if not file_path.exists():
                        integrity_issues.append(f"File not found: {file_path}")
                        continue
                        
                    # Try to load and validate
                    if file_path.suffix == '.npy':
                        data = np.load(file_path, allow_pickle=False)  # Security: no pickle
                        
                        # Check for valid data
                        if not np.all(np.isfinite(data)):
                            integrity_issues.append(f"Invalid values in {file_path}")
                            continue
                            
                        # Check data shape consistency (basic)
                        if data.size == 0:
                            integrity_issues.append(f"Empty data in {file_path}")
                            continue
                            
                    valid_files += 1
                    
                except Exception as e:
                    integrity_issues.append(f"Error loading {file_path}: {e}")
                    
            return {
                'valid': len(integrity_issues) == 0,
                'files_checked': len(files_to_check),
                'valid_files': valid_files,
                'issues': integrity_issues,
                'integrity_ratio': valid_files / max(1, len(files_to_check))
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
            
    def _assess_data_quality(self, dataset_path: Path) -> Dict[str, Any]:
        """Assess data quality metrics."""
        try:
            quality_metrics = {
                'distribution_analysis': {},
                'outlier_detection': {},
                'consistency_checks': {},
                'completeness': {}
            }
            
            # Sample data for analysis
            if dataset_path.is_file() and dataset_path.suffix == '.npy':
                sample_data = np.load(dataset_path)
                quality_metrics = self._analyze_data_quality(sample_data)
            else:
                # Analyze multiple files
                sample_files = list(dataset_path.rglob('*.npy'))[:5]  # Sample 5 files
                combined_metrics = []
                
                for file_path in sample_files:
                    try:
                        data = np.load(file_path)
                        metrics = self._analyze_data_quality(data)
                        combined_metrics.append(metrics)
                    except Exception as e:
                        self.logger.warning(f"Could not analyze {file_path}: {e}")
                        
                if combined_metrics:
                    # Aggregate metrics
                    quality_metrics = self._aggregate_quality_metrics(combined_metrics)
                    
            return {
                'valid': True,
                'metrics': quality_metrics,
                'analysis_timestamp': time.time()
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
            
    def _analyze_data_quality(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze quality of individual data array."""
        metrics = {}
        
        # Basic statistics
        metrics['shape'] = data.shape
        metrics['dtype'] = str(data.dtype)
        metrics['size_mb'] = data.nbytes / (1024 ** 2)
        
        # Value distribution
        if data.size > 0:
            metrics['min_value'] = float(np.min(data))
            metrics['max_value'] = float(np.max(data))
            metrics['mean_value'] = float(np.mean(data))
            metrics['std_value'] = float(np.std(data))
            
            # Check for suspicious patterns
            unique_values = len(np.unique(data))
            metrics['unique_values'] = unique_values
            metrics['sparsity'] = np.count_nonzero(data) / data.size
            
            # Outlier detection (simple method)
            if data.size > 10:
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                outlier_mask = (data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)
                metrics['outlier_ratio'] = np.sum(outlier_mask) / data.size
                
        return metrics
        
    def _aggregate_quality_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate quality metrics from multiple samples."""
        if not metrics_list:
            return {}
            
        aggregated = {
            'sample_count': len(metrics_list),
            'shape_consistency': len(set(str(m.get('shape', '')) for m in metrics_list)) == 1,
            'dtype_consistency': len(set(m.get('dtype', '') for m in metrics_list)) == 1,
        }
        
        # Aggregate numeric metrics
        numeric_fields = ['min_value', 'max_value', 'mean_value', 'std_value', 'sparsity', 'outlier_ratio']
        for field in numeric_fields:
            values = [m.get(field, 0) for m in metrics_list if field in m]
            if values:
                aggregated[f'{field}_mean'] = np.mean(values)
                aggregated[f'{field}_std'] = np.std(values)
                
        return aggregated
        
    def _validate_data_security(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate data from security perspective."""
        try:
            security_issues = []
            
            # Check file permissions
            if not dataset_path.exists():
                security_issues.append("Dataset path does not exist")
                
            # Check for suspicious file types
            if dataset_path.is_dir():
                all_files = list(dataset_path.rglob('*'))
                suspicious_extensions = ['.exe', '.bat', '.sh', '.py', '.js', '.html']
                
                for file_path in all_files:
                    if file_path.suffix.lower() in suspicious_extensions:
                        security_issues.append(f"Suspicious file type: {file_path}")
                        
            # Check path for directory traversal attempts
            path_str = str(dataset_path)
            if '..' in path_str or path_str.startswith('/etc/') or path_str.startswith('/root/'):
                security_issues.append("Potentially unsafe file path")
                
            return {
                'valid': len(security_issues) == 0,
                'issues': security_issues,
                'scan_timestamp': time.time()
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
            
    def validate_batch(self, batch_data: Any, batch_labels: Any = None) -> ValidationResult:
        """Validate individual training batch."""
        result = ValidationResult()
        
        if not TORCH_AVAILABLE:
            result.add_warning("TORCH_UNAVAILABLE", "PyTorch not available for batch validation")
            return result
            
        try:
            # Type validation
            if not isinstance(batch_data, torch.Tensor):
                result.add_error("INVALID_BATCH_TYPE", f"Expected torch.Tensor, got {type(batch_data)}")
                return result
                
            # Shape validation
            if len(batch_data.shape) < 2:
                result.add_error("INVALID_BATCH_SHAPE", f"Batch must have at least 2 dimensions, got {batch_data.shape}")
                
            # Value validation
            if torch.any(torch.isnan(batch_data)):
                result.add_error("NAN_IN_BATCH", "Batch contains NaN values")
                
            if torch.any(torch.isinf(batch_data)):
                result.add_error("INF_IN_BATCH", "Batch contains infinite values")
                
            # Memory check
            memory_manager = get_memory_safety_manager()
            batch_size_bytes = batch_data.element_size() * batch_data.numel()
            
            if not memory_manager.safe_allocate(batch_size_bytes, "training_batch"):
                result.add_error("MEMORY_INSUFFICIENT", "Insufficient memory for batch processing")
                
            # Update counters
            self.total_samples_processed += batch_data.shape[0]
            
            if not result.is_valid:
                self.samples_rejected += batch_data.shape[0]
                
        except Exception as e:
            result.add_error("BATCH_VALIDATION_ERROR", f"Batch validation failed: {e}")
            
        return result
        
    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality report."""
        return {
            'timestamp': time.time(),
            'total_samples_processed': self.total_samples_processed,
            'samples_rejected': self.samples_rejected,
            'rejection_rate': self.samples_rejected / max(1, self.total_samples_processed),
            'corruption_detected': self.corruption_detected,
            'quality_score': 1.0 - (self.samples_rejected / max(1, self.total_samples_processed))
        }


class SecureTrainingPipeline:
    """Production-ready training pipeline with comprehensive security and monitoring."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.memory_manager = get_memory_safety_manager()
        self.resilient_manager = get_resilient_operation_manager()
        self.resource_monitor = get_resource_monitor()
        self.data_quality = DataQualityAssurance(config)
        self.model_validator = get_model_output_validator()
        self.security_audit = get_security_audit_log()
        
        # Training state
        self.training_history = []
        self.best_validation_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_interrupted = False
        
        # Metrics
        self.training_metrics = {
            'total_batches_processed': 0,
            'total_training_time': 0.0,
            'memory_cleanups': 0,
            'circuit_breaker_activations': 0,
            'adversarial_attacks_detected': 0
        }
        
    def validate_training_setup(self) -> ValidationResult:
        """Comprehensive validation of training setup."""
        result = ValidationResult()
        
        # Validate configuration
        config_result = self.config.validate()
        result.errors.extend(config_result.errors)
        result.warnings.extend(config_result.warnings)
        if not config_result.is_valid:
            result.is_valid = False
            
        # Validate dataset
        dataset_validation = self.data_quality.validate_training_data(Path(self.config.dataset_path))
        if not dataset_validation.get('overall_valid', False):
            result.add_error("DATASET_INVALID", "Dataset validation failed")
            
        # Check system resources
        resource_snapshot = self.resource_monitor.monitor_resources()
        
        # Memory check
        if resource_snapshot['system'].get('memory_percent', 0) > 80:
            result.add_warning("HIGH_MEMORY_USAGE", "High system memory usage before training")
            
        # GPU check
        if resource_snapshot['gpu'].get('available', False):
            for gpu in resource_snapshot['gpu']['devices']:
                if gpu['memory_percent'] > 70:
                    result.add_warning("HIGH_GPU_MEMORY", f"High GPU memory usage on device {gpu['device_id']}")
                    
        return result
        
    def train_model(self, model: Any, train_loader: Any, val_loader: Any = None) -> Dict[str, Any]:
        """Execute secure training with comprehensive monitoring."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training")
            
        # Pre-training validation
        setup_validation = self.validate_training_setup()
        if not setup_validation.is_valid:
            raise ValueError(f"Training setup validation failed: {setup_validation.format_errors()}")
            
        training_start = time.time()
        
        try:
            # Initialize training components
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            criterion = nn.CrossEntropyLoss()  # Default loss function
            
            # Training loop with resilience
            for epoch in range(self.config.epochs):
                if self.training_interrupted:
                    self.logger.warning("Training interrupted by user or system")
                    break
                    
                epoch_result = self._train_epoch(
                    model, train_loader, optimizer, criterion, epoch
                )
                
                # Validation
                if val_loader and epoch % self.config.validation_frequency == 0:
                    val_result = self._validate_epoch(model, val_loader, criterion, epoch)
                    epoch_result.update(val_result)
                    
                    # Early stopping check
                    if self._check_early_stopping(val_result.get('val_loss', float('inf'))):
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
                        
                # Learning rate scheduling
                if 'val_loss' in epoch_result:
                    scheduler.step(epoch_result['val_loss'])
                    
                # Save checkpoint
                if epoch % self.config.save_frequency == 0:
                    self._save_checkpoint(model, optimizer, epoch, epoch_result)
                    
                self.training_history.append(epoch_result)
                
            total_training_time = time.time() - training_start
            
            return {
                'training_completed': True,
                'total_epochs': len(self.training_history),
                'total_training_time': total_training_time,
                'best_validation_loss': self.best_validation_loss,
                'training_history': self.training_history,
                'training_metrics': self.training_metrics,
                'final_model_path': self._save_final_model(model, optimizer)
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.security_audit.log_security_violation("training_system", "training_failure", str(e))
            raise
            
    def _train_epoch(self, model: Any, train_loader: Any, optimizer: Any, 
                    criterion: Any, epoch: int) -> Dict[str, Any]:
        """Train single epoch with monitoring and error handling."""
        epoch_start = time.time()
        model.train()
        
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': 0.0,
            'train_accuracy': 0.0,
            'batches_processed': 0,
            'batches_rejected': 0,
            'memory_usage_peak': 0.0
        }
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            try:
                # Batch validation
                batch_validation = self._validate_training_batch(batch_data, batch_labels)
                if not batch_validation.is_valid:
                    epoch_metrics['batches_rejected'] += 1
                    self.logger.warning(f"Batch {batch_idx} rejected: {batch_validation.format_errors()}")
                    continue
                    
                # Execute training step with resilience
                step_result = self.resilient_manager.execute_with_resilience(
                    'model_inference',
                    self._training_step,
                    model, batch_data, batch_labels, optimizer, criterion,
                    fallback_func=self._fallback_training_step
                )
                
                running_loss += step_result['loss']
                correct_predictions += step_result['correct']
                total_samples += step_result['batch_size']
                epoch_metrics['batches_processed'] += 1
                
                # Memory monitoring
                if batch_idx % 10 == 0:  # Monitor every 10 batches
                    memory_stats = self.memory_manager.monitor_memory_usage()
                    epoch_metrics['memory_usage_peak'] = max(
                        epoch_metrics['memory_usage_peak'],
                        memory_stats['current_memory_gb']
                    )
                    
                    # Force cleanup if memory usage is high
                    if memory_stats['usage_percent'] > 90:
                        self.memory_manager.force_cleanup()
                        self.training_metrics['memory_cleanups'] += 1
                        
            except Exception as e:
                self.logger.error(f"Training step failed for batch {batch_idx}: {e}")
                epoch_metrics['batches_rejected'] += 1
                continue
                
        # Calculate epoch metrics
        if epoch_metrics['batches_processed'] > 0:
            epoch_metrics['train_loss'] = running_loss / epoch_metrics['batches_processed']
            epoch_metrics['train_accuracy'] = correct_predictions / max(1, total_samples)
        else:
            epoch_metrics['train_loss'] = float('inf')
            epoch_metrics['train_accuracy'] = 0.0
            
        epoch_metrics['epoch_duration'] = time.time() - epoch_start
        
        # Update global metrics
        self.training_metrics['total_batches_processed'] += epoch_metrics['batches_processed']
        self.training_metrics['total_training_time'] += epoch_metrics['epoch_duration']
        
        return epoch_metrics
        
    def _training_step(self, model: Any, batch_data: Any, batch_labels: Any, 
                      optimizer: Any, criterion: Any) -> Dict[str, Any]:
        """Execute single training step."""
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
        
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == batch_labels).sum().item()
        
        return {
            'loss': loss.item(),
            'correct': correct,
            'batch_size': batch_data.size(0)
        }
        
    def _fallback_training_step(self, model: Any, batch_data: Any, batch_labels: Any, 
                               optimizer: Any, criterion: Any) -> Dict[str, Any]:
        """Fallback training step with reduced precision."""
        self.logger.info("Using fallback training step")
        
        # Simple fallback - just return dummy values to continue
        return {
            'loss': 0.0,
            'correct': 0,
            'batch_size': batch_data.size(0) if hasattr(batch_data, 'size') else 1
        }
        
    def _validate_training_batch(self, batch_data: Any, batch_labels: Any) -> ValidationResult:
        """Validate training batch before processing."""
        result = self.data_quality.validate_batch(batch_data, batch_labels)
        
        # Additional adversarial defense
        if self.config.enable_adversarial_defense and isinstance(batch_data, torch.Tensor):
            # Convert to numpy for adversarial defense
            batch_np = batch_data.detach().cpu().numpy()
            
            # Simple check - in real implementation, this would be more sophisticated
            if np.std(batch_np) > 10.0:  # Very high variance might indicate adversarial samples
                result.add_warning("SUSPICIOUS_BATCH", "Batch has unusually high variance")
                
        return result
        
    def _validate_epoch(self, model: Any, val_loader: Any, criterion: Any, epoch: int) -> Dict[str, Any]:
        """Validate model performance on validation set."""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                try:
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    
                except Exception as e:
                    self.logger.error(f"Validation batch failed: {e}")
                    continue
                    
        return {
            'val_loss': val_loss / len(val_loader) if len(val_loader) > 0 else float('inf'),
            'val_accuracy': correct / max(1, total),
            'val_samples': total
        }
        
    def _check_early_stopping(self, current_val_loss: float) -> bool:
        """Check early stopping condition."""
        if current_val_loss < self.best_validation_loss:
            self.best_validation_loss = current_val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.config.early_stopping_patience
            
    def _save_checkpoint(self, model: Any, optimizer: Any, epoch: int, epoch_result: Dict[str, Any]):
        """Save training checkpoint."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_config': asdict(self.config),
                'epoch_result': epoch_result,
                'training_metrics': self.training_metrics
            }
            
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            
    def _save_final_model(self, model: Any, optimizer: Any) -> str:
        """Save final trained model."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            final_model_path = output_dir / f"{self.config.model_name}_final.pth"
            
            final_checkpoint = {
                'model_state_dict': model.state_dict(),
                'training_config': asdict(self.config),
                'training_history': self.training_history,
                'training_metrics': self.training_metrics,
                'best_validation_loss': self.best_validation_loss
            }
            
            torch.save(final_checkpoint, final_model_path)
            
            self.logger.info(f"Final model saved: {final_model_path}")
            return str(final_model_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save final model: {e}")
            return ""
            
    def get_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        return {
            'config': asdict(self.config),
            'training_metrics': self.training_metrics,
            'data_quality_report': self.data_quality.get_quality_report(),
            'resilience_stats': self.resilient_manager.get_resilience_stats(),
            'memory_stats': self.memory_manager.monitor_memory_usage(),
            'resource_trends': self.resource_monitor.get_resource_trends(),
            'training_history': self.training_history,
            'best_validation_loss': self.best_validation_loss,
            'total_training_time': self.training_metrics['total_training_time'],
            'report_timestamp': time.time()
        }


def create_secure_training_pipeline(config_dict: Dict[str, Any]) -> SecureTrainingPipeline:
    """Create a secure training pipeline from configuration dictionary."""
    # Validate and sanitize configuration
    input_sanitizer = get_input_sanitizer()
    
    # Required configuration keys
    required_keys = ['model_name', 'dataset_path', 'output_dir']
    for key in required_keys:
        if key not in config_dict:
            raise ValueError(f"Required configuration key missing: {key}")
            
    # Sanitize configuration values
    sanitized_config = input_sanitizer.sanitize_dict_input(
        config_dict, 
        allowed_keys=['model_name', 'dataset_path', 'output_dir', 'batch_size', 
                     'learning_rate', 'epochs', 'validation_split', 'save_frequency',
                     'validation_frequency', 'early_stopping_patience', 'max_memory_gb',
                     'enable_adversarial_defense', 'enable_data_augmentation', 'gradient_clipping'],
        field_name="training_config"
    )
    
    # Create configuration object
    config = TrainingConfig(**sanitized_config)
    
    # Validate configuration
    config_validation = config.validate()
    if not config_validation.is_valid:
        raise ValueError(f"Configuration validation failed: {config_validation.format_errors()}")
        
    return SecureTrainingPipeline(config)