# Context Filtering Model
import torch
import torch.nn as nn

class CFM(nn.Module):
    def __init__(self, 
                 d_model) :
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2*d_model),
        )


    def forward(self, x_enc, context_feat) :
        """
        x_enc, context_feat     [B, T, C]

        return 
            fus_feat : [B, T, C]
        """
        B, T, C = x_enc.shape

        avg_x_enc = torch.mean(x_enc, dim=1, keepdim=True)
        avg_context_feat = torch.mean(context_feat, dim=1, keepdim=True)
        weights = self.encode(avg_x_enc + avg_context_feat).reshape(B, 1, C, 2)
        weights = weights.softmax(dim=-2)

        fus_feat = x_enc * weights[..., 0] + context_feat * weights[..., 1]

        return fus_feat
