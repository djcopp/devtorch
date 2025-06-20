import torch
import torch.nn as nn
from ..base import Processor


class FusionProcessor(Processor):
    """Simple concatenation-based fusion processor."""
    
    def __init__(self, input_dim, output_dim=None, dropout=0.1):
        super().__init__()
        
        if output_dim is None:
            output_dim = input_dim
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, *features):
        if len(features) == 1:
            fused = features[0]
        else:
            fused = torch.cat(features, dim=-1)
        
        return self.projection(fused)


class AttentionFusionProcessor(Processor):
    """Attention-based fusion for multiple modalities."""
    
    def __init__(self, input_dims, output_dim=None, num_heads=8, dropout=0.1):
        super().__init__()
        
        if isinstance(input_dims, int):
            input_dims = [input_dims]
        
        self.input_dims = input_dims
        self.feature_dim = max(input_dims)
        
        if output_dim is None:
            output_dim = self.feature_dim
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.feature_dim) if dim != self.feature_dim else nn.Identity()
            for dim in input_dims
        ])
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(self.feature_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(self.feature_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, *features):
        projected_features = []
        
        for feature, projection in zip(features, self.projections):
            if feature.dim() == 2:
                feature = feature.unsqueeze(1)
            projected_features.append(projection(feature))
        
        stacked_features = torch.cat(projected_features, dim=1)
        
        attended, _ = self.attention(stacked_features, stacked_features, stacked_features)
        attended = self.norm(attended)
        
        pooled = torch.mean(attended, dim=1)
        return self.output_projection(pooled)


class BilinearFusionProcessor(Processor):
    """Bilinear fusion for two modalities."""
    
    def __init__(self, input_dim1, input_dim2, output_dim=None, dropout=0.1):
        super().__init__()
        
        if output_dim is None:
            output_dim = min(input_dim1, input_dim2)
        
        self.bilinear = nn.Bilinear(input_dim1, input_dim2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, feature1, feature2):
        fused = self.bilinear(feature1, feature2)
        fused = self.activation(fused)
        return self.dropout(fused)


class GatedFusionProcessor(Processor):
    """Gated fusion mechanism for adaptive feature combination."""
    
    def __init__(self, input_dims, output_dim=None, dropout=0.1):
        super().__init__()
        
        if isinstance(input_dims, int):
            input_dims = [input_dims]
        
        self.num_modalities = len(input_dims)
        total_input_dim = sum(input_dims)
        
        if output_dim is None:
            output_dim = total_input_dim // 2
        
        self.gate_network = nn.Sequential(
            nn.Linear(total_input_dim, self.num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.feature_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ) for dim in input_dims
        ])
    
    def forward(self, *features):
        concatenated = torch.cat(features, dim=-1)
        gates = self.gate_network(concatenated)
        
        processed_features = []
        for i, (feature, network) in enumerate(zip(features, self.feature_networks)):
            processed = network(feature)
            gated = processed * gates[:, i:i+1]
            processed_features.append(gated)
        
        return sum(processed_features)


class CrossModalAttentionProcessor(Processor):
    """Cross-modal attention between two modalities."""
    
    def __init__(self, input_dim1, input_dim2, output_dim=None, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.feature_dim = max(input_dim1, input_dim2)
        
        if output_dim is None:
            output_dim = self.feature_dim
        
        self.proj1 = nn.Linear(input_dim1, self.feature_dim) if input_dim1 != self.feature_dim else nn.Identity()
        self.proj2 = nn.Linear(input_dim2, self.feature_dim) if input_dim2 != self.feature_dim else nn.Identity()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(self.feature_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, feature1, feature2):
        proj1 = self.proj1(feature1)
        proj2 = self.proj2(feature2)
        
        if proj1.dim() == 2:
            proj1 = proj1.unsqueeze(1)
        if proj2.dim() == 2:
            proj2 = proj2.unsqueeze(1)
        
        attended1, _ = self.cross_attention(proj1, proj2, proj2)
        attended2, _ = self.cross_attention(proj2, proj1, proj1)
        
        attended1 = self.norm(attended1).squeeze(1)
        attended2 = self.norm(attended2).squeeze(1)
        
        fused = torch.cat([attended1, attended2], dim=-1)
        return self.fusion(fused)


class HierarchicalFusionProcessor(Processor):
    """Hierarchical fusion for multiple modalities."""
    
    def __init__(self, input_dims, output_dim=None, fusion_strategy='concat', dropout=0.1):
        super().__init__()
        
        if isinstance(input_dims, int):
            input_dims = [input_dims]
        
        self.num_modalities = len(input_dims)
        
        if output_dim is None:
            output_dim = sum(input_dims) // 2
        
        self.pairwise_fusers = nn.ModuleList()
        current_dims = input_dims.copy()
        
        while len(current_dims) > 1:
            new_dims = []
            for i in range(0, len(current_dims), 2):
                if i + 1 < len(current_dims):
                    dim1, dim2 = current_dims[i], current_dims[i + 1]
                    fused_dim = (dim1 + dim2) // 2
                    
                    if fusion_strategy == 'concat':
                        fuser = nn.Sequential(
                            nn.Linear(dim1 + dim2, fused_dim),
                            nn.ReLU(inplace=True),
                            nn.Dropout(dropout)
                        )
                    elif fusion_strategy == 'bilinear':
                        fuser = BilinearFusionProcessor(dim1, dim2, fused_dim, dropout)
                    else:
                        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
                    
                    self.pairwise_fusers.append(fuser)
                    new_dims.append(fused_dim)
                else:
                    new_dims.append(current_dims[i])
            
            current_dims = new_dims
        
        final_dim = current_dims[0]
        self.final_projection = nn.Sequential(
            nn.Linear(final_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        ) if final_dim != output_dim else nn.Identity()
    
    def forward(self, *features):
        current_features = list(features)
        fuser_idx = 0
        
        while len(current_features) > 1:
            next_features = []
            for i in range(0, len(current_features), 2):
                if i + 1 < len(current_features):
                    feature1, feature2 = current_features[i], current_features[i + 1]
                    
                    if isinstance(self.pairwise_fusers[fuser_idx], BilinearFusionProcessor):
                        fused = self.pairwise_fusers[fuser_idx](feature1, feature2)
                    else:
                        concatenated = torch.cat([feature1, feature2], dim=-1)
                        fused = self.pairwise_fusers[fuser_idx](concatenated)
                    
                    next_features.append(fused)
                    fuser_idx += 1
                else:
                    next_features.append(current_features[i])
            
            current_features = next_features
        
        return self.final_projection(current_features[0]) 