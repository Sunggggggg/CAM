# Context Token Modeling
import torch
import torch.nn as nn

from lib.models.DAM import DAM
from lib.models.CAM import CAM
from lib.models.CFM import CFM

class ConTM(nn.Module):
    def __init__(self, 
                 seqlen=16, 
                 d_model=2048, 
                 attn_drop=0.,
                 proj_drop=0.,
                 learnable_alpha=False,
                 ) :
        super().__init__()
        self.dual_attn = DAM(t_dim=d_model, c_dim=seqlen, attn_drop=attn_drop, proj_drop=proj_drop)
        self.context_ext = CAM(seqlen=seqlen, d_model=2048, learnable_alpha=learnable_alpha)
        self.fusing = CFM(d_model)

    def forward(self, x):
        """
        input : [B, T, 2048]
        """
        #x_enc = self.dual_attn(x)               # [B, T, D]
        x_enc = x
        context_feat = self.context_ext(x_enc)  # [B, T, D]
        fusion_feat = self.fusing(x_enc, context_feat)

        return fusion_feat