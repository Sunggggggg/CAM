# Context Accmulation Model
import torch
import torch.nn as nn

from lib.models.DAM import DAM

transpose = lambda x : x.permute(0, 2, 1)

class CAM(nn.Module) :
    def __init__(self, 
                 seqlen=16, 
                 d_model=2048, 
                 num_head=8, 
                 spatial_n_layer=3,
                 attn_drop=0.,
                 proj_drop=0.,
                 ) :
        super().__init__()
        self.dual_attn = DAM(t_dim=d_model, c_dim=seqlen, attn_drop=attn_drop, proj_drop=proj_drop)
    
    
    def forward(self, input, vitpose_j2d=None) :
        """
        input : [B, T, 2048]
        """
        input_enc = self.dual_attn(input)

        
        return 
