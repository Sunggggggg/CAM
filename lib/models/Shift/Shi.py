import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self, ) :
        super().__init__()
        self.cam_proj = nn.Linear(2048, embed_dim//2)

    def forward(self, x):
        """
        x : [B, T, 2048]
        """



        return