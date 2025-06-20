from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms


TRANSFORM_PRESETS = {
    "imagenet_basic": {
        "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        "description": "Basic ImageNet preprocessing"
    },
    
    "imagenet_train": {
        "transform": transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        "description": "ImageNet training transforms with augmentation"
    },
    
    "imagenet_val": {
        "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        "description": "ImageNet validation transforms"
    },
    
    "cifar_train": {
        "transform": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ]),
        "description": "CIFAR training transforms"
    },
    
    "cifar_test": {
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ]),
        "description": "CIFAR test/validation transforms"
    },
    
    "high_res_train": {
        "transform": transforms.Compose([
            transforms.RandomResizedCrop((512, 512), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        "description": "High resolution training transforms (512x512)"
    },
    
    "high_res_val": {
        "transform": transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.CenterCrop((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        "description": "High resolution validation transforms (512x512)"
    },
    
    "heavy_augment": {
        "transform": transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        "description": "Heavy augmentation for robust training"
    },
    
    "minimal": {
        "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        "description": "Minimal processing - just resize and convert to tensor"
    }
}


class CustomCompose:
    """
    Custom compose class that provides more flexibility than torchvision.transforms.Compose.
    
    Allows for conditional transforms and better debugging.
    """
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, img):
        for transform in self.transforms:
            try:
                img = transform(img)
            except Exception as e:
                print(f"Error applying transform {transform}: {e}")
                raise
        return img
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class ConditionalTransform:
    """
    Apply a transform conditionally based on a probability or condition.
    
    Args:
        transform: Transform to apply
        condition: Either a float (probability) or callable (condition function)
    """
    
    def __init__(self, transform: Callable, condition: Union[float, Callable]):
        self.transform = transform
        self.condition = condition
        
        if isinstance(condition, float):
            if not 0 <= condition <= 1:
                raise ValueError("Probability must be between 0 and 1")
            self.condition_fn = lambda x: torch.rand(1).item() < condition
        else:
            self.condition_fn = condition
    
    def __call__(self, img):
        if self.condition_fn(img):
            return self.transform(img)
        return img


class MultiScaleResize:
    """
    Resize to multiple scales randomly during training.
    
    Useful for training models that need to handle different input sizes.
    """
    
    def __init__(self, sizes: List[Union[int, Tuple[int, int]]], mode: str = "random"):
        self.sizes = sizes
        self.mode = mode
        
        # Convert int sizes to tuples
        self.sizes = [size if isinstance(size, tuple) else (size, size) for size in sizes]
    
    def __call__(self, img):
        if self.mode == "random":
            size = self.sizes[torch.randint(0, len(self.sizes), (1,)).item()]
        elif self.mode == "largest":
            size = max(self.sizes, key=lambda x: x[0] * x[1])
        elif self.mode == "smallest":
            size = min(self.sizes, key=lambda x: x[0] * x[1])
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return transforms.Resize(size)(img)



def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy for classification."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total


def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Calculate top-k accuracy for classification."""
    _, predicted = torch.topk(outputs, k, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(predicted)
    correct = (predicted == targets_expanded).sum().item()
    total = targets.size(0)
    return correct / total


COMMON_METRICS = {
    "accuracy": accuracy,
    "top5_accuracy": lambda outputs, targets: top_k_accuracy(outputs, targets, k=5)
} 