import torch
import torch.nn as nn
from lib.models.transformer import ED_Transformer


"""Warpping Module"""
class WM(nn.Module):
    def __init__(self,
                 seqlen=16,
                 cam_layer_depth=2,
                 po_sh_layer_depth=3,
                 embed_dim=256,
                 num_head=8,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 attn_drop_rate=0.0,
                 ) :
        super().__init__()
        self.embed_dim = embed_dim
        self.cam_proj = nn.Linear(2048, embed_dim//2)
        self.cam_enc_dec = ED_Transformer(depth=cam_layer_depth, embed_dim=embed_dim//2, mlp_hidden_dim=embed_dim*2, 
                                       h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                                       attn_drop_rate=attn_drop_rate, length=seqlen)

        self.pose_shape_proj = nn.Linear(2048, embed_dim)
        self.pose_shape_encoder = ED_Transformer(depth=po_sh_layer_depth, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*2, 
                                       h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                                       attn_drop_rate=attn_drop_rate, length=seqlen)

    def foward_refine(self, x):
        B, T = x.shape[:2]
        x = self.pose_shape_proj(x)
        
        c_feats = []
        for t in range(1, T) :
            c_feat_a = x[:, t, :self.embed_dim//2]        # [B, 256]
            p_feat_b = x[:, t-1, self.embed_dim//2:]      # [B, 256]

            c_feat = torch.cat([c_feat_a, p_feat_b], dim=-1)
            c_feats.append(c_feat)
        c_feats = [c_feats[0]] + c_feats                  # padding
        c_feats = torch.stack(c_feats, dim=1)             # [B, T, 512]

        return 

    def forward(self, x) :
        """
        Input 
            x : [B, T, 2048]
        """
        ##########################
        # Camera parameter 
        ##########################
        cam_feat = self.cam_proj(x)                 # [B, T, d]
        cam_feat = self.cam_enc_dec(cam_feat)       # [B, T, 128]

        ##########################
        # Camera parameter 
        ##########################
        
        

        # Transformer
        # transformer(c_feats)
        
        return c_feats