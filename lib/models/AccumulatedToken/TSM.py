"""Temporal shift module"""
import torch
import torch.nn as nn

class TSM(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(embed_dim*2),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def foward_refine(self, x):
        """Past -> Future"""
        B, T = x.shape[:2]
        
        c_feats = []
        for t in range(1, T) :
            c_feat_a = x[:, t, :self.embed_dim//2]        # [B, 128]
            p_feat_b = x[:, t-1, self.embed_dim//2:]      # [B, 128]

            c_feat = torch.cat([c_feat_a, p_feat_b], dim=-1)
            c_feats.append(c_feat)
        c_feats = [c_feats[0]] + c_feats                  # padding
        c_feats = torch.stack(c_feats, dim=1)             # [B, T, 256]

        c_feats = c_feats + self.ffn(c_feats)
        c_feats = self.norm(c_feats)

        return c_feats
    
    def backward_refine(self, x):
        """Future -> Past"""
        B, T = x.shape[:2]
        
        c_feats = []
        for t in range(T-1, 0, -1) :
            p_feat_b = x[:, t, self.embed_dim//2:]       
            c_feat_a = x[:, t-1, :self.embed_dim//2]      

            c_feat = torch.cat([c_feat_a, p_feat_b], dim=-1)
            c_feats.append(c_feat)
        c_feats = [c_feats[0]] + c_feats                  # padding
        c_feats = torch.stack(c_feats, dim=1)            

        c_feats = c_feats + self.ffn(c_feats)
        c_feats = self.norm(c_feats)

        return c_feats
