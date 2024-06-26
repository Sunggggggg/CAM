import numpy as np
import torch
import torch.nn as nn

from lib.models.CAM import CAM
from lib.models.CFM import CFM
from lib.models.AccumulatedToken.enc_dec import ED_Transformer
from lib.models.AccumulatedToken.Regressor import CamRegressor, Regressor, regressor_output, Total_Regressor
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.models.AccumulatedToken.Fusion import FusingBlock   

"""Accumulated Token Module"""
class ATM(nn.Module):
    def __init__(self,
                 seqlen=16,
                 cam_layer_depth=3,
                 po_sh_layer_depth=3,
                 embed_dim=256,
                 num_head=8,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 attn_drop_rate=0.0,
                 ) :
        super().__init__()
        self.seqlen = seqlen
        ##########################
        # Camera parameter 
        ##########################
        self.cam_shape_proj = nn.Linear(2048, embed_dim)
        self.cam_shape_enc_dec = ED_Transformer(depth=cam_layer_depth, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*2, 
                                       h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                                       attn_drop_rate=attn_drop_rate, length=seqlen)

        ##########################
        # Accumulated Token
        ##########################
        self.pose_proj = nn.Linear(2048, embed_dim//2)
        self.context_tokenizer = CAM(seqlen=seqlen, d_model=embed_dim//2, d_token=embed_dim//4)
        self.pose_enc_dec = ED_Transformer(depth=po_sh_layer_depth, embed_dim=embed_dim//2, mlp_hidden_dim=embed_dim, 
                                       h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                                       attn_drop_rate=attn_drop_rate, length=seqlen)
        self.fusing = CFM(embed_dim//2)
        self.regressor = Total_Regressor(embed_dim//2 + embed_dim)

        ##########################
        # SMPL
        ##########################
        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)

    def forward(self, x, is_train=False, J_regressor=None) :
        """
        Input 
            x : [B, T, 2048]
        Return
            'theta'  : [B, T, 85]
            'verts'  : [B, T, 6890, 3]
            'kp_2d'  : [B, T, 49, 2]
            'kp_3d'  : [B, T, 49, 3]
            'rotmat' : [B, T, 24, 3, 3]
        """
        ##########################
        # Camera parameter 
        ##########################
        cam_shape_feat = self.cam_shape_proj(x)                 # [B, T, d]
        cam_shape_feat = self.cam_shape_enc_dec(cam_shape_feat)       # [B, T, 128]

        ##########################
        # Accumulated Token
        ##########################
        pose_feat = self.pose_proj(x)                       # [B, T, 512]
        context_feat = self.context_tokenizer(pose_feat)    # [B, T, 512]
        pose_feat = self.fusing(pose_feat, context_feat) 
        pose_feat = self.pose_enc_dec(pose_feat)

        ##########################
        # Regressor
        ##########################
        if is_train :
            size = self.seqlen
        else :
            size = 1
            mid_frame = self.seqlen // 2
            cam_shape_feat = cam_shape_feat[:, mid_frame:mid_frame+1]       # [B, 1, d]
            pose_feat = pose_feat[:, mid_frame:mid_frame+1]         # [B, 1, d]
        
        # pred_cam = self.regressor_cam(cam_feat)                 # [B, T, 3]
        # pred_pose, pred_shape = self.regressor(ps_feat)         #
        feat = torch.cat([cam_shape_feat, pose_feat], dim=-1)
        pred_pose, pred_shape, pred_cam = self.regressor(feat)

        output = regressor_output(self.smpl, pred_pose, pred_shape, pred_cam, size, J_regressor=J_regressor)

        return output