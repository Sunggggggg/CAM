import torch
import torch.nn as nn

from lib.models.DST.GMM import GMM
from lib.models.DST.regressor import Regressor
from lib.models.trans_operator import Attention

"""Decoupling Spatial Temporal modeling"""
class DST(nn.Module):
    def __init__(self,
                 seqlen=16,
                 d_model=512,
                 n_layers=3,
                 num_head=8, 
                 dropout=0., 
                 drop_path_r=0., 
                 atten_drop=0.,
                 mask_ratio=0.,
                 ):
        super().__init__()
        self.seqlen = seqlen
        ##########################
        # Temporal
        ##########################
        self.temporal_modeling = GMM(seqlen, n_layers, d_model, num_head, 
                                   dropout, drop_path_r, atten_drop, mask_ratio)
        ##########################
        # Spatial
        ##########################
        self.s_proj = nn.Linear(2048, d_model)
        self.spatial_modeling = Attention(d_model, num_heads=num_head, attn_drop=atten_drop, proj_drop=dropout)

        ##########################
        # Spatial
        ##########################
        self.regressor = Regressor(d_model+d_model//2)

    def forward(self, x, is_train=False, J_regressor=None):
        """
        x : [B, T, D]
        """
        B = x.shape[0]
        
        smpl_output_global, mask_ids, pred_temp, pred_global = self.temporal_modeling(x, is_train=is_train, J_regressor=J_regressor)    # [B, L, D]
        
        pred_spat = self.s_proj(x[:, self.seqlen//2])[:, None, :]
        pred_spat = self.spatial_modeling(pred_spat)

        pred_temp = pred_temp[:, self.seqlen//2][:, None, :]
        feature = torch.cat([pred_spat, pred_temp], dim=-1)    # []

        smpl_output = self.regressor(feature, init_pose=pred_global[0][:, self.seqlen//2 : self.seqlen//2+1],
                                      init_shape=pred_global[1][:, self.seqlen//2 : self.seqlen//2+1], 
                                      init_cam=pred_global[2][:, self.seqlen//2 : self.seqlen//2+1], is_train=is_train, J_regressor=J_regressor)
        
        scores = None
        if not is_train:    # Eval
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)         
                s['verts'] = s['verts'].reshape(B, -1, 3)      
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)
                s['scores'] = scores

        else:
            size = 1
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)           # [B, 3, 10]
                s['verts'] = s['verts'].reshape(B, size, -1, 3)        # [B, 3, 6980]
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)        # [B, 3, 2]
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)        # [B, 3, 3]
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)   # [B, 3, 3, 3]
                s['scores'] = scores

        return smpl_output, mask_ids, smpl_output_global

