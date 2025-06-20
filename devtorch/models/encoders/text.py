import torch
import torch.nn as nn
from ..base import Encoder


class TextEncoder(Encoder):
    """Simple text encoder using embeddings and LSTM."""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, output_dim=512, 
                 num_layers=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self._output_dim = output_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
            lstm_out = lstm_out * mask_expanded
            sum_hidden = torch.sum(lstm_out, dim=1)
            seq_len = torch.sum(mask_expanded, dim=1)
            pooled = sum_hidden / seq_len
        else:
            pooled = torch.mean(lstm_out, dim=1)
        
        output = self.projection(pooled)
        return output
    
    @property
    def output_dim(self):
        return self._output_dim


class BERTEncoder(Encoder):
    """BERT-based text encoder (requires transformers library)."""
    
    def __init__(self, model_name='bert-base-uncased', output_dim=768, freeze_bert=False):
        super().__init__()
        self._output_dim = output_dim
        
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("transformers library is required for BERTEncoder. Install with: pip install transformers")
        
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        if output_dim != bert_dim:
            self.projection = nn.Sequential(
                nn.Linear(bert_dim, output_dim),
                nn.Tanh()
            )
        else:
            self.projection = nn.Identity()
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.projection(pooled_output)
    
    @property
    def output_dim(self):
        return self._output_dim


class TransformerEncoder(Encoder):
    """Custom transformer encoder for text."""
    
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, 
                 max_length=512, output_dim=512, dropout=0.1):
        super().__init__()
        self._output_dim = output_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_length)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.Tanh()
        )
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = self.pos_encoding(embedded)
        embedded = self.dropout(embedded)
        
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        transformer_out = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(transformer_out.size()).float()
            transformer_out = transformer_out * mask_expanded
            sum_hidden = torch.sum(transformer_out, dim=1)
            seq_len = torch.sum(mask_expanded, dim=1)
            pooled = sum_hidden / seq_len
        else:
            pooled = torch.mean(transformer_out, dim=1)
        
        output = self.projection(pooled)
        return output
    
    @property
    def output_dim(self):
        return self._output_dim


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, embed_dim, max_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class RNNEncoder(Encoder):
    """Generic RNN encoder (LSTM/GRU) for text."""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, output_dim=512,
                 num_layers=2, rnn_type='LSTM', dropout=0.1, bidirectional=True):
        super().__init__()
        self._output_dim = output_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                embed_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                embed_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.projection = nn.Sequential(
            nn.Linear(rnn_output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        rnn_out, _ = self.rnn(embedded)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(rnn_out.size()).float()
            rnn_out = rnn_out * mask_expanded
            sum_hidden = torch.sum(rnn_out, dim=1)
            seq_len = torch.sum(mask_expanded, dim=1)
            pooled = sum_hidden / seq_len
        else:
            pooled = torch.mean(rnn_out, dim=1)
        
        output = self.projection(pooled)
        return output
    
    @property
    def output_dim(self):
        return self._output_dim 