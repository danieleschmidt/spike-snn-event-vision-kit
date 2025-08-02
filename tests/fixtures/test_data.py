"""Test data fixtures and generators for spike-snn-event-vision-kit."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import tempfile
from pathlib import Path


class TestDataGenerator:
    """Generate synthetic test data for various components."""
    
    @staticmethod
    def generate_events(
        width: int = 128,
        height: int = 128,
        duration: float = 1.0,
        event_rate: float = 1000,
        noise_level: float = 0.1,
        seed: Optional[int] = 42
    ) -> np.ndarray:
        """Generate synthetic event stream.
        
        Args:
            width: Sensor width in pixels
            height: Sensor height in pixels  
            duration: Duration in seconds
            event_rate: Events per second
            noise_level: Fraction of noise events (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            np.ndarray: Events in format (x, y, timestamp, polarity)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_events = int(duration * event_rate)
        
        # Generate structured events (moving objects)
        signal_events = int(n_events * (1 - noise_level))
        noise_events = n_events - signal_events
        
        # Signal events: moving horizontal bar
        signal_x = []
        signal_y = []
        signal_t = []
        signal_p = []
        
        for i in range(signal_events):
            t = i * duration / signal_events
            # Horizontal bar moving downward
            y_pos = int((t / duration) * height) % height
            x_pos = np.random.randint(0, width)
            
            signal_x.append(x_pos)
            signal_y.append(y_pos)
            signal_t.append(t)
            signal_p.append(1)  # ON events
        
        # Noise events
        noise_x = np.random.randint(0, width, size=noise_events)
        noise_y = np.random.randint(0, height, size=noise_events)
        noise_t = np.sort(np.random.uniform(0, duration, size=noise_events))
        noise_p = np.random.choice([0, 1], size=noise_events)
        
        # Combine and sort by timestamp
        all_x = np.concatenate([signal_x, noise_x])
        all_y = np.concatenate([signal_y, noise_y])
        all_t = np.concatenate([signal_t, noise_t])
        all_p = np.concatenate([signal_p, noise_p])
        
        # Sort by timestamp
        sort_idx = np.argsort(all_t)
        
        return np.column_stack([
            all_x[sort_idx],
            all_y[sort_idx], 
            all_t[sort_idx],
            all_p[sort_idx]
        ])
    
    @staticmethod
    def generate_spike_trains(
        batch_size: int = 4,
        channels: int = 2,
        height: int = 32,
        width: int = 32,
        time_steps: int = 10,
        sparsity: float = 0.1,
        seed: Optional[int] = 42
    ) -> torch.Tensor:
        """Generate synthetic spike trains.
        
        Args:
            batch_size: Number of samples in batch
            channels: Number of channels (e.g., ON/OFF polarity)
            height: Spatial height dimension
            width: Spatial width dimension
            time_steps: Number of time steps
            sparsity: Fraction of active spikes (0-1)
            seed: Random seed
            
        Returns:
            torch.Tensor: Spike trains of shape (batch, channels, height, width, time)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate random spikes with specified sparsity
        spike_prob = torch.rand(batch_size, channels, height, width, time_steps)
        spikes = (spike_prob < sparsity).float()
        
        return spikes
    
    @staticmethod
    def generate_detection_labels(
        n_samples: int = 100,
        n_classes: int = 10,
        max_objects_per_image: int = 3,
        image_size: Tuple[int, int] = (128, 128),
        seed: Optional[int] = 42
    ) -> List[Dict]:
        """Generate detection labels for object detection tasks.
        
        Args:
            n_samples: Number of samples
            n_classes: Number of object classes
            max_objects_per_image: Maximum objects per image
            image_size: Image dimensions (height, width)
            seed: Random seed
            
        Returns:
            List of detection dictionaries with boxes, labels, scores
        """
        if seed is not None:
            np.random.seed(seed)
        
        height, width = image_size
        labels = []
        
        for _ in range(n_samples):
            n_objects = np.random.randint(1, max_objects_per_image + 1)
            
            boxes = []
            class_ids = []
            scores = []
            
            for _ in range(n_objects):
                # Generate random bounding box
                x1 = np.random.randint(0, width // 2)
                y1 = np.random.randint(0, height // 2)
                x2 = np.random.randint(x1 + 10, width)
                y2 = np.random.randint(y1 + 10, height)
                
                boxes.append([x1, y1, x2, y2])
                class_ids.append(np.random.randint(0, n_classes))
                scores.append(np.random.uniform(0.5, 1.0))
            
            labels.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(class_ids, dtype=torch.long),
                'scores': torch.tensor(scores, dtype=torch.float32)
            })
        
        return labels
    
    @staticmethod
    def generate_neuromorphic_dataset(
        dataset_name: str = "test_dataset",
        n_samples: int = 100,
        save_dir: Optional[Path] = None
    ) -> Dict:
        """Generate a complete neuromorphic dataset.
        
        Args:
            dataset_name: Name of the dataset
            n_samples: Number of samples to generate
            save_dir: Directory to save dataset files
            
        Returns:
            Dictionary with dataset metadata and file paths
        """
        if save_dir is None:
            save_dir = Path(tempfile.mkdtemp())
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_info = {
            'name': dataset_name,
            'n_samples': n_samples,
            'save_dir': str(save_dir),
            'files': []
        }
        
        # Generate event files and labels
        for i in range(n_samples):
            # Generate events
            events = TestDataGenerator.generate_events(
                width=128, height=128, duration=1.0, seed=i
            )
            
            # Save events as .npy file
            event_file = save_dir / f"events_{i:04d}.npy"
            np.save(event_file, events)
            
            # Generate corresponding labels
            labels = TestDataGenerator.generate_detection_labels(
                n_samples=1, seed=i
            )[0]
            
            # Save labels
            label_file = save_dir / f"labels_{i:04d}.pt"
            torch.save(labels, label_file)
            
            dataset_info['files'].append({
                'events': str(event_file),
                'labels': str(label_file)
            })
        
        # Save dataset metadata
        metadata_file = save_dir / 'metadata.pt'
        torch.save(dataset_info, metadata_file)
        
        return dataset_info


class PerformanceTestData:
    """Generate data for performance testing and benchmarking."""
    
    @staticmethod
    def get_benchmark_configs() -> List[Dict]:
        """Get standard benchmark configurations."""
        return [
            {
                'name': 'small_model_light_load',
                'batch_size': 1,
                'input_size': (64, 64),
                'time_steps': 5,
                'n_events': 1000
            },
            {
                'name': 'medium_model_normal_load', 
                'batch_size': 4,
                'input_size': (128, 128),
                'time_steps': 10,
                'n_events': 10000
            },
            {
                'name': 'large_model_heavy_load',
                'batch_size': 8,
                'input_size': (256, 256),
                'time_steps': 20,
                'n_events': 50000
            }
        ]
    
    @staticmethod
    def generate_stress_test_events(
        n_events: int = 100000,
        burst_intensity: float = 10.0,
        seed: Optional[int] = 42
    ) -> np.ndarray:
        """Generate high-intensity event bursts for stress testing.
        
        Args:
            n_events: Total number of events
            burst_intensity: Intensity multiplier for burst regions
            seed: Random seed
            
        Returns:
            np.ndarray: High-intensity event stream
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate base events
        base_events = TestDataGenerator.generate_events(
            width=256, height=256, duration=1.0, 
            event_rate=n_events, seed=seed
        )
        
        # Create burst regions (20% of time with 10x intensity)
        burst_mask = np.random.random(len(base_events)) < 0.2
        burst_events = base_events[burst_mask]
        
        # Duplicate burst events to increase intensity
        multiplied_bursts = []
        for _ in range(int(burst_intensity)):
            noise_offset = np.random.uniform(-1e-6, 1e-6, len(burst_events))
            burst_copy = burst_events.copy()
            burst_copy[:, 2] += noise_offset  # Add small time noise
            multiplied_bursts.append(burst_copy)
        
        # Combine all events
        all_events = np.vstack([base_events] + multiplied_bursts)
        
        # Sort by timestamp
        sort_idx = np.argsort(all_events[:, 2])
        
        return all_events[sort_idx]


class MockHardwareData:
    """Mock hardware interfaces for testing."""
    
    @staticmethod
    def mock_loihi_response() -> Dict:
        """Mock Intel Loihi hardware response."""
        return {
            'status': 'success',
            'latency_us': 150,
            'power_mw': 0.5,
            'spike_count': np.random.randint(100, 1000),
            'core_utilization': np.random.uniform(0.1, 0.9)
        }
    
    @staticmethod
    def mock_akida_response() -> Dict:
        """Mock BrainChip Akida hardware response."""
        return {
            'status': 'success',
            'latency_us': 800,
            'power_mw': 2.1,
            'inference_count': 1,
            'memory_usage_mb': np.random.uniform(10, 50)
        }
    
    @staticmethod
    def mock_event_camera_stream(n_frames: int = 10) -> List[np.ndarray]:
        """Mock event camera data stream."""
        frames = []
        for i in range(n_frames):
            events = TestDataGenerator.generate_events(
                duration=0.1, event_rate=5000, seed=i
            )
            frames.append(events)
        return frames


# Export main classes
__all__ = [
    'TestDataGenerator',
    'PerformanceTestData', 
    'MockHardwareData'
]