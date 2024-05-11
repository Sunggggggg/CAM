
import torch
import torch.nn as nn

# Dual Attention Module
class DAM(nn.Module):
    def __init__(self, t_dim, c_dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.) :
        super().__init__()
        self.t_scale = qk_scale or t_dim ** -0.5
        self.c_scale = qk_scale or c_dim ** -0.5

        # Temporal-axis
        self.t_qkv = nn.Linear(t_dim, t_dim * 3, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)
        self.t_proj = nn.Linear(t_dim, t_dim)
        self.t_proj_drop = nn.Dropout(proj_drop)

        # Channel-axis
        self.c_qkv = nn.Linear(c_dim, c_dim * 3, bias=qkv_bias)
        self.c_attn_drop = nn.Dropout(attn_drop)
        self.c_proj = nn.Linear(c_dim, c_dim)
        self.c_proj_drop = nn.Dropout(proj_drop)

        # Alpha
        self.alpha = nn.Parameter(torch.zeros(2))
    
    def temporal_attn(self, x):
        """
        x   : [B, T, C]
        """
        B, T, C = x.shape
        qkv = self.t_qkv(x).reshape(B, T, 3, C).permute(2, 0, 1, -1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.t_scale
        attn = attn.softmax(dim=-1)
        attn = self.t_attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.t_proj(x)
        x = self.t_proj_drop(x)

        return x

    def channel_atten(self, x):
        """
        x   : [B, T, C]
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        B, C, T = x.shape
        qkv = self.c_qkv(x).reshape(B, C, 3, T).permute(2, 0, 1, -1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.c_scale
        attn = attn.softmax(dim=-1)
        attn = self.c_attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.c_proj(x)
        x = self.c_proj_drop(x)

        x = x.permute(0, 2, 1) # [B, T, C]
        return x
    
    def forward(self, x):
        """
        x : [B, T, D]
        
        return :
            fus_enc : [B, T, D]
        """
        temp_enc = self.temporal_attn(x)
        chan_enc = self.channel_atten(x)

        fus_enc = self.alpha[0] * temp_enc + self.alpha[1] * chan_enc
        return fus_enc
