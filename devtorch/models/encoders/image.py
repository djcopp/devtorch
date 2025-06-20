import torch
import torch.nn as nn
import torchvision.models as models
from ..base import Encoder


class ImageEncoder(Encoder):
    """Generic CNN encoder for image data."""
    
    def __init__(self, input_channels=3, output_dim=512, hidden_dims=[64, 128, 256]):
        super().__init__()
        self._output_dim = output_dim
        
        layers = []
        in_channels = input_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    @property
    def output_dim(self):
        return self._output_dim


class ResNetEncoder(Encoder):
    """ResNet-based encoder with pretrained weights."""
    
    def __init__(self, model_name='resnet50', pretrained=True, output_dim=512, freeze_backbone=False):
        super().__init__()
        self._output_dim = output_dim
        
        if model_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif model_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif model_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif model_name == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x
    
    @property
    def output_dim(self):
        return self._output_dim


class EfficientNetEncoder(Encoder):
    """EfficientNet-based encoder."""
    
    def __init__(self, model_name='efficientnet_b0', pretrained=True, output_dim=512, freeze_backbone=False):
        super().__init__()
        self._output_dim = output_dim
        
        if model_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained=pretrained)
            backbone_dim = 1280
        elif model_name == 'efficientnet_b1':
            backbone = models.efficientnet_b1(pretrained=pretrained)
            backbone_dim = 1280
        elif model_name == 'efficientnet_b2':
            backbone = models.efficientnet_b2(pretrained=pretrained)
            backbone_dim = 1408
        else:
            raise ValueError(f"Unsupported EfficientNet model: {model_name}")
        
        self.backbone = backbone.features
        self.avgpool = backbone.avgpool
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(backbone_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.projection(x)
        return x
    
    @property
    def output_dim(self):
        return self._output_dim


class CNNEncoder(Encoder):
    """Simple CNN encoder for basic image classification."""
    
    def __init__(self, input_channels=3, output_dim=512, num_blocks=4):
        super().__init__()
        self._output_dim = output_dim
        
        channels = [input_channels] + [64 * (2**i) for i in range(num_blocks)]
        
        layers = []
        for i in range(num_blocks):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
        
        self.features = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(channels[-1], output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.projection(x)
        return x
    
    @property
    def output_dim(self):
        return self._output_dim 