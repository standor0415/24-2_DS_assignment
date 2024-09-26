import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        #TODO
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        # Multi-head attention
        self.multihead_attn = MultiHeadAttention(d_model, n_heads)
        # Feed-forward
        self.ff = FeedForwardLayer(d_model, d_ff)
        
        # Normalization and Dropout
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.dropout3 = DropoutLayer(dropout)
        
        # Residual connections
        self.res1 = ResidualConnection()
        self.res2 = ResidualConnection()
        self.res3 = ResidualConnection()
    
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        self_attn_output = self.dropout1(self_attn_output)
        x = self.res1(x, self_attn_output)
        x = self.norm1(x)

        # Multi-head attention
        multihead_attn_output = self.multihead_attn(x, memory, memory, src_mask)
        multihead_attn_output = self.dropout2(multihead_attn_output)
        x = self.res2(x, multihead_attn_output)
        x = self.norm2(x)

        # 3. Feed-forward 
        ff_output = self.ff(x)
        ff_output = self.dropout3(ff_output)
        x = self.res3(x, ff_output)
        x = self.norm3(x)

        return x