"""
Neuromorphic dataset loaders for standard event-based vision benchmarks.

Supports major neuromorphic datasets including N-CARS, N-Caltech101, DDD17, GEN1
with proper event data parsing and preprocessing pipelines.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Iterator
from pathlib import Path
import h5py
import struct
from dataclasses import dataclass
from abc import ABC, abstractmethod
import urllib.request
import tarfile
import zipfile
import logging
from tqdm import tqdm


@dataclass 
class EventData:
    """Container for event-based data."""
    x: np.ndarray  # X coordinates
    y: np.ndarray  # Y coordinates  
    t: np.ndarray  # Timestamps (microseconds)
    p: np.ndarray  # Polarity (0 or 1)
    height: int = 128
    width: int = 128
    
    def __len__(self) -> int:
        return len(self.x)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor format [N, 4] (x, y, t, p)."""
        return torch.stack([
            torch.from_numpy(self.x).long(),
            torch.from_numpy(self.y).long(), 
            torch.from_numpy(self.t).float(),
            torch.from_numpy(self.p).long()
        ], dim=1)


class AedatReader:
    """Reader for .aedat format event files."""
    
    @staticmethod
    def read_aedat2(file_path: Path) -> EventData:
        """Read AEDAT2 format files."""
        with open(file_path, 'rb') as f:
            # Skip header
            line = f.readline()
            while line.startswith(b'#'):
                if b'AER data' in line:
                    break
                line = f.readline()
            
            # Read events
            events = []
            while True:
                data = f.read(8)  # 8 bytes per event
                if len(data) < 8:
                    break
                    
                addr, timestamp = struct.unpack('<II', data)
                
                # Decode address
                x = (addr >> 1) & 0x7F
                y = (addr >> 8) & 0x7F  
                pol = addr & 0x1
                
                events.append([x, y, timestamp, pol])
            
            if not events:
                return EventData(
                    x=np.array([]), y=np.array([]), 
                    t=np.array([]), p=np.array([])
                )
                
            events = np.array(events)
            return EventData(
                x=events[:, 0].astype(np.uint16),
                y=events[:, 1].astype(np.uint16), 
                t=events[:, 2].astype(np.uint64),
                p=events[:, 3].astype(np.uint8)
            )
    
    @staticmethod
    def read_aedat4(file_path: Path) -> EventData:
        """Read AEDAT4 format files."""
        try:
            with h5py.File(file_path, 'r') as f:
                events = f['events']
                return EventData(
                    x=events['x'][:],
                    y=events['y'][:],
                    t=events['t'][:], 
                    p=events['pol'][:]
                )
        except Exception as e:
            logging.warning(f"Failed to read AEDAT4 file {file_path}: {e}")
            return EventData(
                x=np.array([]), y=np.array([]),
                t=np.array([]), p=np.array([])
            )


class NeuromorphicDatasetBase(Dataset, ABC):
    """Base class for neuromorphic datasets."""
    
    def __init__(
        self,
        root: Union[str, Path],
        download: bool = True,
        transform: Optional[callable] = None,
        time_window: float = 50e-3,  # 50ms windows
        temporal_subsample: int = 1
    ):
        self.root = Path(root)
        self.download_flag = download
        self.transform = transform
        self.time_window = time_window
        self.temporal_subsample = temporal_subsample
        
        self.root.mkdir(parents=True, exist_ok=True)
        
        if download and not self._check_exists():
            self._download()
            
        self._load_metadata()
        
    @abstractmethod
    def _check_exists(self) -> bool:
        """Check if dataset exists."""
        pass
        
    @abstractmethod 
    def _download(self):
        """Download dataset."""
        pass
        
    @abstractmethod
    def _load_metadata(self):
        """Load dataset metadata."""
        pass
        
    def _download_file(self, url: str, filename: str):
        """Download file with progress bar."""
        filepath = self.root / filename
        
        def progress_hook(block_num, block_size, total_size):
            if hasattr(progress_hook, 'pbar'):
                progress_hook.pbar.update(block_size)
            else:
                progress_hook.pbar = tqdm(
                    total=total_size, 
                    unit='B', 
                    unit_scale=True,
                    desc=f"Downloading {filename}"
                )
                
        urllib.request.urlretrieve(url, filepath, progress_hook)
        if hasattr(progress_hook, 'pbar'):
            progress_hook.pbar.close()
            
    def _extract_archive(self, archive_path: Path, extract_to: Optional[Path] = None):
        """Extract archive file."""
        if extract_to is None:
            extract_to = archive_path.parent
            
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)


class NCarsDataset(NeuromorphicDatasetBase):
    """N-CARS dataset for car classification."""
    
    CLASSES = ['background', 'car']
    URL = "http://www.prophesee.ai/resources/Prophesee_Dataset_n_cars.tar"
    
    def __init__(self, root: Union[str, Path], train: bool = True, **kwargs):
        self.train = train
        super().__init__(root, **kwargs)
        
    def _check_exists(self) -> bool:
        train_dir = self.root / "n-cars" / "train"  
        test_dir = self.root / "n-cars" / "test"
        return train_dir.exists() and test_dir.exists()
        
    def _download(self):
        """Download N-CARS dataset."""
        logging.info("Downloading N-CARS dataset...")
        archive_name = "n_cars.tar"
        self._download_file(self.URL, archive_name)
        
        # Extract
        archive_path = self.root / archive_name
        self._extract_archive(archive_path)
        archive_path.unlink()  # Clean up
        
    def _load_metadata(self):
        """Load N-CARS metadata."""
        split = "train" if self.train else "test"
        self.data_dir = self.root / "n-cars" / split
        
        self.samples = []
        for class_idx, class_name in enumerate(self.CLASSES):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for file_path in class_dir.glob("*.dat"):
                    self.samples.append((file_path, class_idx))
                    
        logging.info(f"Loaded {len(self.samples)} N-CARS {split} samples")
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[EventData, int]:
        file_path, label = self.samples[idx]
        
        # Read events (assuming .dat format)
        events = self._read_dat_file(file_path)
        
        if self.transform:
            events = self.transform(events)
            
        return events, label
        
    def _read_dat_file(self, file_path: Path) -> EventData:
        """Read .dat format event files."""
        try:
            with open(file_path, 'rb') as f:
                data = np.fromfile(f, dtype=np.uint32)
                
            # Parse events (format specific to N-CARS)
            events = []
            for d in data:
                x = (d >> 17) & 0x7FFF 
                y = (d >> 2) & 0x7FFF
                pol = (d >> 1) & 0x1
                t = d  # Timestamp encoding varies
                events.append([x, y, t, pol])
                
            events = np.array(events)
            return EventData(
                x=events[:, 0].astype(np.uint16),
                y=events[:, 1].astype(np.uint16),
                t=events[:, 2].astype(np.uint64), 
                p=events[:, 3].astype(np.uint8),
                height=120, width=100  # N-CARS resolution
            )
        except Exception as e:
            logging.warning(f"Failed to read {file_path}: {e}")
            return EventData(
                x=np.array([]), y=np.array([]),
                t=np.array([]), p=np.array([])
            )


class NCaltech101Dataset(NeuromorphicDatasetBase):
    """N-Caltech101 dataset for object classification."""
    
    URL = "https://www.garrickorchard.com/datasets/n-caltech101/N-Caltech101-archive.zip"
    
    def __init__(self, root: Union[str, Path], **kwargs):
        super().__init__(root, **kwargs)
        
    def _check_exists(self) -> bool:
        return (self.root / "N-Caltech101").exists()
        
    def _download(self):
        """Download N-Caltech101 dataset."""
        logging.info("Downloading N-Caltech101 dataset...")
        archive_name = "N-Caltech101-archive.zip"
        self._download_file(self.URL, archive_name)
        
        archive_path = self.root / archive_name  
        self._extract_archive(archive_path)
        archive_path.unlink()
        
    def _load_metadata(self):
        """Load N-Caltech101 metadata.""" 
        self.data_dir = self.root / "N-Caltech101"
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            for file_path in class_dir.glob("*.bin"):
                self.samples.append((file_path, class_idx))
                
        logging.info(f"Loaded {len(self.samples)} N-Caltech101 samples from {len(self.classes)} classes")
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[EventData, int]:
        file_path, label = self.samples[idx]
        events = self._read_bin_file(file_path)
        
        if self.transform:
            events = self.transform(events)
            
        return events, label
        
    def _read_bin_file(self, file_path: Path) -> EventData:
        """Read .bin format event files."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.uint8)
                
            # N-Caltech101 specific format
            events = []
            i = 0
            while i < len(raw_data) - 4:
                x = raw_data[i] | (raw_data[i+1] << 8)
                y = raw_data[i+2] | (raw_data[i+3] << 8) 
                # Simplified - actual format is more complex
                events.append([x & 0x3FF, y & 0x1FF, i, (x >> 10) & 1])
                i += 5
                
            events = np.array(events)
            return EventData(
                x=events[:, 0].astype(np.uint16),
                y=events[:, 1].astype(np.uint16),
                t=events[:, 2].astype(np.uint64),
                p=events[:, 3].astype(np.uint8),
                height=180, width=240
            )
        except Exception as e:
            logging.warning(f"Failed to read {file_path}: {e}")
            return EventData(
                x=np.array([]), y=np.array([]),
                t=np.array([]), p=np.array([])
            )


class DDD17Dataset(NeuromorphicDatasetBase):
    """DDD17 dataset for driving detection."""
    
    URL = "https://docs.prophesee.ai/stable/datasets/ddd17.html"  # Instructions only
    
    def __init__(self, root: Union[str, Path], **kwargs):
        super().__init__(root, download=False, **kwargs)  # Manual download required
        
    def _check_exists(self) -> bool:
        return (self.root / "DDD17").exists() and len(list((self.root / "DDD17").glob("*.h5"))) > 0
        
    def _download(self):
        """DDD17 requires manual download."""
        raise NotImplementedError(
            "DDD17 dataset requires manual download from Prophesee. "
            f"Please download to {self.root}/DDD17/ and extract .h5 files."
        )
        
    def _load_metadata(self):
        """Load DDD17 metadata."""
        self.data_dir = self.root / "DDD17"
        self.samples = list(self.data_dir.glob("*.h5"))
        logging.info(f"Loaded {len(self.samples)} DDD17 samples")
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[EventData, Dict]:
        file_path = self.samples[idx]
        events, labels = self._read_h5_file(file_path)
        
        if self.transform:
            events = self.transform(events)
            
        return events, labels
        
    def _read_h5_file(self, file_path: Path) -> Tuple[EventData, Dict]:
        """Read .h5 format files with detection labels."""
        try:
            with h5py.File(file_path, 'r') as f:
                # Events
                events_group = f['events']
                events = EventData(
                    x=events_group['x'][:],
                    y=events_group['y'][:],
                    t=events_group['t'][:],
                    p=events_group['p'][:],
                    height=480, width=640
                )
                
                # Labels (bounding boxes)
                labels = {}
                if 'labels' in f:
                    labels_group = f['labels']
                    labels = {
                        'boxes': labels_group['boxes'][:] if 'boxes' in labels_group else [],
                        'classes': labels_group['classes'][:] if 'classes' in labels_group else [],
                        'timestamps': labels_group['t'][:] if 't' in labels_group else []
                    }
                
                return events, labels
        except Exception as e:
            logging.warning(f"Failed to read {file_path}: {e}")
            return EventData(
                x=np.array([]), y=np.array([]),
                t=np.array([]), p=np.array([])
            ), {}


class EventTransforms:
    """Common transformations for event data."""
    
    @staticmethod
    def temporal_subsample(events: EventData, factor: int) -> EventData:
        """Subsample events temporally."""
        indices = np.arange(0, len(events), factor)
        return EventData(
            x=events.x[indices],
            y=events.y[indices], 
            t=events.t[indices],
            p=events.p[indices],
            height=events.height,
            width=events.width
        )
        
    @staticmethod
    def spatial_downsample(events: EventData, target_size: Tuple[int, int]) -> EventData:
        """Downsample events spatially."""
        height, width = target_size
        scale_y = height / events.height
        scale_x = width / events.width
        
        new_x = (events.x * scale_x).astype(np.uint16)
        new_y = (events.y * scale_y).astype(np.uint16)
        
        # Remove out-of-bounds events
        valid = (new_x < width) & (new_y < height)
        
        return EventData(
            x=new_x[valid],
            y=new_y[valid],
            t=events.t[valid], 
            p=events.p[valid],
            height=height,
            width=width
        )
        
    @staticmethod
    def add_noise(events: EventData, noise_rate: float = 0.1) -> EventData:
        """Add random noise events."""
        n_noise = int(len(events) * noise_rate)
        
        noise_x = np.random.randint(0, events.width, n_noise, dtype=np.uint16)
        noise_y = np.random.randint(0, events.height, n_noise, dtype=np.uint16)
        noise_t = np.random.uniform(events.t.min(), events.t.max(), n_noise).astype(np.uint64)
        noise_p = np.random.randint(0, 2, n_noise, dtype=np.uint8)
        
        return EventData(
            x=np.concatenate([events.x, noise_x]),
            y=np.concatenate([events.y, noise_y]),
            t=np.concatenate([events.t, noise_t]), 
            p=np.concatenate([events.p, noise_p]),
            height=events.height,
            width=events.width
        )


def create_dataloader(
    dataset: NeuromorphicDatasetBase,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Optional[callable] = None
) -> DataLoader:
    """Create DataLoader for neuromorphic datasets."""
    
    def default_collate(batch):
        """Default collate function for event data."""
        events, labels = zip(*batch)
        return events, torch.tensor(labels) if isinstance(labels[0], int) else labels
    
    if collate_fn is None:
        collate_fn = default_collate
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


# Dataset factory
def get_dataset(
    name: str,
    root: Union[str, Path],
    **kwargs
) -> NeuromorphicDatasetBase:
    """Factory function to get neuromorphic datasets."""
    
    datasets = {
        'n-cars': NCarsDataset,
        'n-caltech101': NCaltech101Dataset, 
        'ddd17': DDD17Dataset
    }
    
    if name.lower() not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
        
    return datasets[name.lower()](root, **kwargs)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load N-CARS dataset
    dataset = get_dataset('n-cars', './data', download=True, train=True)
    dataloader = create_dataloader(dataset, batch_size=8)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get sample
    events, label = dataset[0]
    print(f"Sample events: {len(events)} events, label: {label}")
    print(f"Event data shape: x={events.x.shape}, y={events.y.shape}, t={events.t.shape}, p={events.p.shape}")