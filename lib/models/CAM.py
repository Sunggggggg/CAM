# Context Accmulation Model
import torch
import torch.nn as nn

from lib.models.DAM import DAM

transpose = lambda x : x.permute(0, 2, 1)

class CAM(nn.Module) :
    def __init__(self,
                 seqlen=16,
                 d_model=2048,
                 learnable_alpha=False 
                 ) :
        super().__init__()
        self.seqlen = seqlen
        self.learnable_alpha = learnable_alpha
        assert d_model % seqlen == 0, f"{d_model} % {seqlen} = 0" 

        if learnable_alpha :
            self.alpha = nn.Parameter(torch.ones(1, 1, seqlen-1) * 0.5, requires_grad=False)
        else :
            self.alpha = nn.Parameter(torch.randn(1, 1, seqlen-1))

        d_token = int(d_model // seqlen)
        self.split = nn.Linear(d_model, d_token)
        self.norm = nn.LayerNorm(d_token)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x_enc) :
        """
        x_enc : [B, T, D]
        """
        split_x_enc = self.split(x_enc)         # [B, T, D/T]
        split_x_enc = self.norm(split_x_enc)
        
        for t in range(self.seqlen) :
            if t == 0 :
            
            else :
            
            split_x_enc[:, t]
        

        return 
