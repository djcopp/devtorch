import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """
    Simple dataset wrapper for when you have data already in memory.
    
    Args:
        data: List of data samples or (input, target) tuples
        transform: Optional transform to apply to inputs
        target_transform: Optional transform to apply to targets
    """
    
    def __init__(
        self,
        data: List[Any],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Any:
        sample = self.data[idx]
        
        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            input_data, target = sample
            
            if self.transform:
                input_data = self.transform(input_data)
            
            if self.target_transform:
                target = self.target_transform(target)
            
            return input_data, target
        else:
            if self.transform:
                sample = self.transform(sample)
            return sample


class ImageDataset(Dataset):
    """
    Dataset for loading images from a CSV file or directory structure.
    
    Supports both single-label and multi-label classification.
    
    Args:
        csv_file: Path to CSV file containing image paths and labels
        root_dir: Root directory containing images (used with csv_file)
        image_column: Name of column containing image paths (default: 'image_path')
        label_columns: Name(s) of label columns (default: 'label')
        transform: Optional transform to apply to images
        multi_label: Whether this is a multi-label classification task
    """
    
    def __init__(
        self,
        csv_file: str,
        root_dir: Optional[str] = None,
        image_column: str = 'image_path',
        label_columns: Union[str, List[str]] = 'label',
        transform: Optional[Callable] = None,
        multi_label: bool = False
    ):
        self.csv_file = csv_file
        self.root_dir = Path(root_dir) if root_dir else None
        self.image_column = image_column
        self.label_columns = label_columns if isinstance(label_columns, list) else [label_columns]
        self.transform = transform
        self.multi_label = multi_label
        
        # Load CSV data
        self.data = pd.read_csv(csv_file)
        
        # Validate columns exist
        if image_column not in self.data.columns:
            raise ValueError(f"Image column '{image_column}' not found in CSV")
        
        for label_col in self.label_columns:
            if label_col not in self.data.columns:
                raise ValueError(f"Label column '{label_col}' not found in CSV")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        row = self.data.iloc[idx]
        
        # Load image
        image_path = row[self.image_column]
        if self.root_dir:
            image_path = self.root_dir / image_path
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        if len(self.label_columns) == 1:
            label = row[self.label_columns[0]]
            if isinstance(label, str) and self.multi_label:
                # Handle comma-separated multi-labels
                labels = [l.strip() for l in label.split(',')]
                label = torch.zeros(len(self.get_unique_labels()))
                for l in labels:
                    if l in self.label_to_idx:
                        label[self.label_to_idx[l]] = 1
            else:
                label = torch.tensor(label, dtype=torch.long)
        else:
            # Multi-column labels (for multi-head models)
            label = {}
            for label_col in self.label_columns:
                label[label_col] = torch.tensor(row[label_col], dtype=torch.long)
        
        return image, label
    
    def get_unique_labels(self) -> List[str]:
        """Get unique labels for multi-label classification."""
        if not hasattr(self, '_unique_labels'):
            all_labels = set()
            for _, row in self.data.iterrows():
                label = row[self.label_columns[0]]
                if isinstance(label, str):
                    labels = [l.strip() for l in label.split(',')]
                    all_labels.update(labels)
                else:
                    all_labels.add(str(label))
            
            self._unique_labels = sorted(list(all_labels))
            self.label_to_idx = {label: idx for idx, label in enumerate(self._unique_labels)}
        
        return self._unique_labels


class FolderDataset(Dataset):
    """
    Dataset for loading images from folder structure.
    
    Expects folder structure like:
    root/
    ├── class1/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── class2/
        ├── img3.jpg
        └── img4.jpg
    
    Args:
        root_dir: Root directory containing class folders
        transform: Optional transform to apply to images
        extensions: Valid image extensions (default: common image formats)
        class_to_idx: Optional mapping from class names to indices
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tiff', '.tif'),
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        
        # Find all classes and images
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        if class_to_idx is None:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
        
        # Find all images
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in self.extensions:
                    self.samples.append((str(image_path), class_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class MultiDataset(Dataset):
    """
    Dataset that combines multiple datasets.
    
    Useful for combining different data sources or creating multi-task datasets.
    
    Args:
        datasets: Dictionary mapping dataset names to Dataset instances  
        sampling_strategy: How to sample from datasets ('round_robin', 'proportional', 'equal')
    """
    
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        sampling_strategy: str = 'proportional'
    ):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        self.sampling_strategy = sampling_strategy
        
        # Calculate dataset sizes
        self.dataset_sizes = {name: len(dataset) for name, dataset in datasets.items()}
        self.total_size = sum(self.dataset_sizes.values())
        
        # Create sampling indices based on strategy
        self._create_sampling_indices()
    
    def _create_sampling_indices(self):
        """Create indices for sampling based on strategy."""
        if self.sampling_strategy == 'round_robin':
            # Sample evenly from each dataset
            max_size = max(self.dataset_sizes.values())
            self.sampling_plan = []
            
            for i in range(max_size):
                for dataset_name in self.dataset_names:
                    if i < self.dataset_sizes[dataset_name]:
                        self.sampling_plan.append((dataset_name, i))
        
        elif self.sampling_strategy == 'proportional':
            # Sample proportionally to dataset sizes
            self.sampling_plan = []
            for dataset_name, dataset_size in self.dataset_sizes.items():
                for i in range(dataset_size):
                    self.sampling_plan.append((dataset_name, i))
        
        elif self.sampling_strategy == 'equal':
            # Sample equally from each dataset
            min_size = min(self.dataset_sizes.values())
            self.sampling_plan = []
            
            for i in range(min_size):
                for dataset_name in self.dataset_names:
                    self.sampling_plan.append((dataset_name, i))
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
    
    def __len__(self) -> int:
        return len(self.sampling_plan)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        dataset_name, dataset_idx = self.sampling_plan[idx]
        dataset = self.datasets[dataset_name]
        
        sample = dataset[dataset_idx]
        
        # Return sample with dataset name for multi-task learning
        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            inputs, targets = sample
        else:
            inputs, targets = sample, None
        
        return inputs, {'targets': targets, 'dataset': dataset_name}


def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset into training and validation sets.
    
    Args:
        dataset: Dataset to split
        val_ratio: Fraction of data to use for validation
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    dataset_size = len(dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    return train_dataset, val_dataset


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True
) -> Union[torch.utils.data.DataLoader, Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]:
    """
    Create data loaders from datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle_train: Whether to shuffle training data
        
    Returns:
        DataLoader or tuple of (train_loader, val_loader)
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return train_loader, val_loader
    
    return train_loader 