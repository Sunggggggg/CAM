import numpy as np
import torch
import torch.nn as nn

from lib.models.transformer_global import Transformer
from lib.models.AccumulatedToken.enc_dec import ED_Transformer
from lib.models.AccumulatedToken.Regressor import CamRegressor, Regressor, regressor_output, Total_Regressor
from lib.models.AccumulatedToken.Fusion import FusingBlock   

"""Accumulated Token Module"""
class ATM(nn.Module):
    def __init__(self,
                 seqlen=16,
                 cam_layer_depth=2,
                 po_sh_layer_depth=3,
                 embed_dim=512,
                 num_head=8,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 attn_drop_rate=0.0,
                 ) :
        super().__init__()
        self.seqlen = seqlen
        ##########################
        # Encoder
        ##########################
        self.input_proj = nn.Linear(2048, embed_dim)
        self.global_encoder = Transformer(depth=2, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*2, 
                    h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                    attn_drop_rate=attn_drop_rate, length=seqlen)
        
        ##########################
        # Accumulate Concept
        ##########################

        ##########################
        # Decoder
        ##########################

    def forward(self, x, is_train=False, J_regressor=None) :
        """
        Input 
            x : [B, T, 2048]
        Return
            'theta'  : [B, T, 85]
            'verts'  : [B, T, 6890, 3]
            'kp_2d'  : [B, T, 49, 2]
            'kp_3d'  : [B, T, 49, 3]
            'rotmat' : [B, T, 24, 3, 3]
        """
        ##########################
        # Encoder
        ##########################
        x = self.input_proj(x)                                              # [B, T, 512]
        x = self.global_encoder(x, is_train=is_train, mask_ratio=0.5)       # [B, T, 256]




        