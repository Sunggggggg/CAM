import torch
import torch.nn as nn


# Context Token Generation Moduel
class CTG(nn.Module) :
    def __init__(self, d_model=2048, d_token=256) :
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_token),
            nn.LayerNorm(d_token)
        )

    def forward(self, x):
        """
        x : [B, T, C]
        """
        B, T = x.shape[:2]

        abs_diff = torch.stack([torch.abs(x[:, t+1] - x[:, t]) for t in range(T - 1)], dim=1)  # [B, C]
        acc_feat = torch.mean(abs_diff, dim=1)

        context_token = self.proj(acc_feat)         # [B, T-1, D]



        return