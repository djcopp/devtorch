import torch
import torch.nn as nn
from ..base import Decoder


class ClassificationDecoder(Decoder):
    """Simple classification decoder with optional intermediate layers."""
    
    def __init__(self, input_dim, num_classes, hidden_dims=None, dropout=0.1, activation='relu'):
        super().__init__(input_dim, num_classes)
        
        if hidden_dims is None:
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, num_classes))
            self.classifier = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        return self.classifier(x)


class MultiLabelClassificationDecoder(Decoder):
    """Multi-label classification decoder with sigmoid activation."""
    
    def __init__(self, input_dim, num_classes, hidden_dims=None, dropout=0.1):
        super().__init__(input_dim, num_classes)
        
        if hidden_dims is None:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, num_classes),
                nn.Sigmoid()
            )
        else:
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.extend([
                nn.Linear(prev_dim, num_classes),
                nn.Sigmoid()
            ])
            self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


class HierarchicalClassificationDecoder(Decoder):
    """Hierarchical classification with multiple levels."""
    
    def __init__(self, input_dim, level_classes, hidden_dim=None, dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension
            level_classes: List of number of classes at each level [level1_classes, level2_classes, ...]
            hidden_dim: Hidden dimension for intermediate processing
            dropout: Dropout rate
        """
        super().__init__(input_dim, sum(level_classes))
        
        self.level_classes = level_classes
        self.num_levels = len(level_classes)
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.level_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) 
            for num_classes in level_classes
        ])
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        
        outputs = {}
        for i, classifier in enumerate(self.level_classifiers):
            outputs[f'level_{i+1}'] = classifier(shared_features)
        
        return outputs


class AttentionClassificationDecoder(Decoder):
    """Classification decoder with attention mechanism."""
    
    def __init__(self, input_dim, num_classes, num_heads=8, dropout=0.1):
        super().__init__(input_dim, num_classes)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        attended, _ = self.attention(x, x, x)
        attended = self.norm(attended + x)
        
        pooled = torch.mean(attended, dim=1)
        return self.classifier(pooled)


class EnsembleClassificationDecoder(Decoder):
    """Ensemble of multiple classification heads."""
    
    def __init__(self, input_dim, num_classes, num_heads=3, hidden_dim=None, dropout=0.1):
        super().__init__(input_dim, num_classes)
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
            for _ in range(num_heads)
        ])
        
        self.num_heads = num_heads
    
    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        
        stacked = torch.stack(outputs, dim=0)
        return torch.mean(stacked, dim=0)


class PrototypicalClassificationDecoder(Decoder):
    """Prototypical networks for few-shot classification."""
    
    def __init__(self, input_dim, num_classes, distance_metric='euclidean'):
        super().__init__(input_dim, num_classes)
        
        self.prototypes = nn.Parameter(torch.randn(num_classes, input_dim))
        self.distance_metric = distance_metric
        
        nn.init.xavier_uniform_(self.prototypes)
    
    def forward(self, x):
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(x, self.prototypes, p=2)
        elif self.distance_metric == 'cosine':
            x_norm = torch.nn.functional.normalize(x, dim=1)
            proto_norm = torch.nn.functional.normalize(self.prototypes, dim=1)
            distances = 1 - torch.mm(x_norm, proto_norm.t())
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        return -distances 