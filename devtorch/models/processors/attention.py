import torch
import torch.nn as nn
from ..base import Processor


class AttentionProcessor(Processor):
    """Multi-head self-attention processor."""
    
    def __init__(self, input_dim, num_heads=8, dropout=0.1, use_layer_norm=True):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(input_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        attended, _ = self.attention(x, x, x)
        
        if self.use_layer_norm:
            attended = self.norm(attended + x)
        
        attended = self.dropout(attended)
        
        return attended.squeeze(1) if attended.size(1) == 1 else attended


class SelfAttentionProcessor(Processor):
    """Simple self-attention mechanism."""
    
    def __init__(self, input_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 4
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.scale = hidden_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        B, L, D = x.shape
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, v)
        output = self.norm(attended + x)
        
        return output.squeeze(1) if output.size(1) == 1 else output


class CrossAttentionProcessor(Processor):
    """Cross-attention between query and key-value pairs."""
    
    def __init__(self, query_dim, kv_dim, output_dim=None, num_heads=8, dropout=0.1):
        super().__init__()
        
        if output_dim is None:
            output_dim = query_dim
        
        self.query_projection = nn.Linear(query_dim, output_dim)
        self.kv_projection = nn.Linear(kv_dim, output_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value):
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)
        
        q = self.query_projection(query)
        kv = self.kv_projection(key_value)
        
        attended, attention_weights = self.attention(q, kv, kv)
        attended = self.norm(attended + q)
        attended = self.dropout(attended)
        
        output = attended.squeeze(1) if attended.size(1) == 1 else attended
        return output, attention_weights


class PositionalAttentionProcessor(Processor):
    """Attention processor with positional encoding."""
    
    def __init__(self, input_dim, max_length=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(input_dim, max_length)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.pos_encoding(x)
        
        attended, _ = self.attention(x, x, x)
        attended = self.norm(attended + x)
        attended = self.dropout(attended)
        
        return attended.squeeze(1) if attended.size(1) == 1 else attended


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-style attention."""
    
    def __init__(self, embed_dim, max_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LocalAttentionProcessor(Processor):
    """Local attention with limited window size."""
    
    def __init__(self, input_dim, window_size=32, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        B, L, D = x.shape
        
        if L <= self.window_size:
            attended, _ = self.attention(x, x, x)
            attended = self.norm(attended + x)
            return self.dropout(attended).squeeze(1) if attended.size(1) == 1 else self.dropout(attended)
        
        outputs = []
        for i in range(0, L, self.window_size):
            end_idx = min(i + self.window_size, L)
            window = x[:, i:end_idx, :]
            
            attended, _ = self.attention(window, window, window)
            attended = self.norm(attended + window)
            outputs.append(attended)
        
        output = torch.cat(outputs, dim=1)
        output = self.dropout(output)
        
        return output.squeeze(1) if output.size(1) == 1 else output


class SparseAttentionProcessor(Processor):
    """Sparse attention with configurable sparsity pattern."""
    
    def __init__(self, input_dim, block_size=32, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.block_size = block_size
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        B, L, D = x.shape
        
        num_blocks = (L + self.block_size - 1) // self.block_size
        padded_length = num_blocks * self.block_size
        
        if L < padded_length:
            padding = torch.zeros(B, padded_length - L, D, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x
        
        x_blocks = x_padded.view(B, num_blocks, self.block_size, D)
        x_blocks = x_blocks.view(B * num_blocks, self.block_size, D)
        
        attended, _ = self.attention(x_blocks, x_blocks, x_blocks)
        attended = self.norm(attended + x_blocks)
        
        attended = attended.view(B, num_blocks, self.block_size, D)
        attended = attended.view(B, padded_length, D)
        
        if L < padded_length:
            attended = attended[:, :L, :]
        
        output = self.dropout(attended)
        return output.squeeze(1) if output.size(1) == 1 else output 