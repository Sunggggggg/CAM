import torch
import torch.nn as nn

# Context Token Generation Moduel
class CTG(nn.Module) :
    def __init__(self, d_model, d_token, abs_flag=False) :
        super().__init__()
        self.abs_flag = abs_flag

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_token),
            nn.LayerNorm(d_token)
        )
    
    def forward(self, x):
        """
        x : [B, C, T]
        """
        B, C = x.shape[:2]

        if self.abs_flag : 
            diff = torch.stack([torch.abs(x[:, t+1] - x[:, t]) for t in range(C - 1)], dim=1)  # [B, C]
        else :
            diff = torch.stack([(x[:, t+1] - x[:, t]) for t in range(C - 1)], dim=1)  # [B, C]
        
        acc_feat = torch.mean(diff, dim=1)
        context_token = self.proj(acc_feat)         # [B, T-1, D]



        return