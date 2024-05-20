import numpy as np
import torch
import torch.nn as nn

from lib.models.transformer import Transformer
from lib.models.AccumulatedToken.TSM import TSM
from lib.models.AccumulatedToken.Regressor import Total_Regressor, regressor_output
from lib.models.smpl import SMPL, SMPL_MODEL_DIR

"""Accumulated Token Module"""
class ATM(nn.Module):
    def __init__(self,
                 seqlen=16,
                 cam_layer_depth=2,
                 po_sh_layer_depth=3,
                 embed_dim=512,
                 num_head=8,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 attn_drop_rate=0.0,
                 ) :
        super().__init__()
        self.seqlen = seqlen
        ##########################
        # Encoder
        ##########################
        self.input_proj = nn.Linear(2048, embed_dim)
        self.global_encoder = Transformer(depth=2, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*2, 
                    h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                    attn_drop_rate=attn_drop_rate, length=seqlen)
        
        ##########################
        # Accumulate Concept
        ##########################
        self.short_embed_dim = embed_dim // 2

        ##########################
        # Camera 
        ##########################
        cam_embed_dim = embed_dim // 4
        self.cam_proj = nn.Linear(embed_dim, cam_embed_dim)
        self.cam_norm = nn.LayerNorm(cam_embed_dim)

        self.cam_dec = Transformer(depth=1, embed_dim=cam_embed_dim, mlp_hidden_dim=cam_embed_dim*2, 
                    h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                    attn_drop_rate=attn_drop_rate, length=seqlen)

        ##########################
        # Pose, Shape 
        ##########################
        pose_shape_embed_dim = embed_dim // 2
        self.pose_shape_proj = nn.Linear(embed_dim, pose_shape_embed_dim)
        self.pose_shape_norm = nn.LayerNorm(pose_shape_embed_dim)

        self.tsm = TSM(pose_shape_embed_dim)
        self.pose_shape_dec = Transformer(depth=2, embed_dim=pose_shape_embed_dim, mlp_hidden_dim=pose_shape_embed_dim*2, 
                    h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                    attn_drop_rate=attn_drop_rate, length=seqlen)
        
        self.apply(self._init_weights)

        ##########################
        # SMPL
        ##########################
        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)
        self.regressor = Total_Regressor(pose_shape_embed_dim+cam_embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
        # Encoder
        ##########################
        x = self.input_proj(x)          # [B, T, 512]
        x = self.global_encoder(x)      # [B, T, 512]

        ##########################
        # Camera 
        ##########################
        cam_feat = self.cam_proj(x)         # [B, T, 128]
        cam_feat = self.cam_norm(cam_feat)  # [B, T, 128]
        cam_feat = self.cam_dec(cam_feat)

        ##########################
        # Pose, Shape 
        ##########################
        pose_shape_feat = self.pose_shape_proj(x)   # [B, T, 256]
        pose_shape_feat = self.pose_shape_norm(pose_shape_feat)

        pose_shape_feat = self.tsm.foward_refine(pose_shape_feat)
        pose_shape_feat = self.tsm.backward_refine(pose_shape_feat) # [B, T, 256]
        pose_shape_feat = self.pose_shape_dec(pose_shape_feat)

        ##########################
        # Regressor
        ##########################
        if is_train :
            size = self.seqlen
        else :
            size = 1
            mid_frame = self.seqlen // 2
            cam_feat = cam_feat[:, mid_frame:mid_frame+1]       # [B, 1, d]
            ps_feat = ps_feat[:, mid_frame:mid_frame+1]         # [B, 1, d]

        feature = torch.cat([pose_shape_feat, cam_feat], dim=-1)    # [B, T, 256+128]
        pred_pose, pred_shape, pred_cam = self.regressor(feature)

        output = regressor_output(self.smpl, pred_pose, pred_shape, pred_cam, size, J_regressor=J_regressor)

        return output