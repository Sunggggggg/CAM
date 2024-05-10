import torch
import torch.nn as nn

accumulate = lambda x, y, alpha : (x + alpha * y) / (1 + alpha)

# Context Token Generation Module
class CTG(nn.Module) :
    def __init__(self, seqlen=16, d_model=2048, d_token=256) :
        super().__init__()
        assert d_model % seqlen == 0, "Checkout CTG channel"
        self.seqlen = seqlen

    def forward(self, x):
        """
        x : [B, T, C]
        """
        B, T, C = x.shape[0]
        
        

        return