import numpy as np
import torch
import torch.nn as nn

from lib.models.CAM import CAM
from lib.models.CFM import CFM
from lib.models.AccumulatedToken.enc_dec import ED_Transformer
from lib.models.AccumulatedToken.Regressor import CamRegressor, Regressor, regressor_output

"""Accumulated Token Module"""
class ATM(nn.Module):
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
        ##########################
        # Camera parameter 
        ##########################
        self.cam_proj = nn.Linear(2048, embed_dim//2)
        self.cam_enc_dec = ED_Transformer(depth=cam_layer_depth, embed_dim=embed_dim//2, mlp_hidden_dim=embed_dim*2, 
                                       h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                                       attn_drop_rate=attn_drop_rate, length=seqlen)
        self.regressor_cam = CamRegressor(d_model=embed_dim//2)

        ##########################
        # Accumulated Token
        ##########################
        self.context_tokenizer = CAM(seqlen=seqlen, d_model=2048, d_token=embed_dim)
        self.pose_shape_encoder = ED_Transformer(depth=po_sh_layer_depth, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*2, 
                                       h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                                       attn_drop_rate=attn_drop_rate, length=seqlen)
        self.fusing = CFM(embed_dim)
        self.regressor = Regressor(embed_dim)

    def forward(self, x, J_regressor=None) :
        """
        x : [B, T, 2048]
        """
        ##########################
        # Camera parameter 
        ##########################
        cam_feat = self.cam_proj(x)                 # [B, T, d]
        cam_feat = self.cam_enc_dec(cam_feat)       # [B, T, d]
        pred_cam = self.regressor_cam(cam_feat)     # [B, T, 3]

        ##########################
        # Accumulated Token
        ##########################
        context_feat = self.context_tokenizer(x)        # [B, T, 256]
        x = self.fusing(x, context_feat)
        ps_feat = self.pose_shape_encoder(x)
        pred_pose, pred_shape = self.regressor(ps_feat) #

        ##########################
        # Output
        ##########################
        output = regressor_output(pred_pose, pred_shape, pred_cam, J_regressor=J_regressor)