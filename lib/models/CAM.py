# Context Accmulation Model
import torch
import torch.nn as nn

from lib.models.CTG import CTG
from lib.models.transformer import Transformer

class CAM(nn.Module) :
    def __init__(self, seqlen=16, d_model=2048, num_head=8, spatial_n_layer=3) :
        super().__init__()
        self.spatial_attn = Transformer(depth=spatial_n_layer, embed_dim=seqlen, mlp_hidden_dim=seqlen*2, h=num_head, length=d_model)
        
    def forward(self, x) :
        """
        
        """
        x = x.permute(0, 2, 1)
        x = self.spatial_attn(x)
        
        return x