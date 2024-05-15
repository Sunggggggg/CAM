import torch
import torch.nn as nn


""""""
class FusingBlock(nn.Module):
    def __init__(self, d_model, qkv_bias=False, qk_scale=None) :
        super().__init__()

        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.kv_proj = nn.Linear(d_model, d_model*2, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model)

    def crossatttn(self, x, y):
        B, T, D = x.shape

        q = self.q_proj(x)
        kv = self.kv_proj(y).reshape(B, T, D, 2)
        k, v = kv[..., 0], kv[..., 1]

        refine_q = []
        for t in range(T) :
            attn = q[:, t:t+1] @ k[:, t:t+1].transpose(-2, -1)  # [B, 1, 1]
            attn = attn.softmax(dim=-1)
            x = attn @ v[:, t:t+1]
            x = self.proj(x)        # [B, 1, d]
            refine_q.append(x)
        refine_q = torch.cat(refine_q, dim=1)

        return refine_q

    def forward(self, x, y) :
        """
        Input
            x, y : [B, T, D]
        """
        fus_feat = self.crossatttn(x, y)

        return fus_feat
        
