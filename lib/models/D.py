import numpy as np
import torch
import torch.nn as nn

from lib.models.CAM import CAM
from lib.models.CFM import CFM
from lib.models.transformer import Transformer
from lib.models.smpl import SMPL_MEAN_PARAMS


class CamRegressor(nn.Module) :
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS) :
        super().__init__()
        mean_params = np.load(smpl_mean_params)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_cam', init_cam)  # 3

        self.fc1 = nn.Linear(3+256, 256)
        self.drop1 = nn.Dropout()
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

    def forward(self, x, n_iter=3) :
        B = x.shape[0]
        x = x.reshape(-1, x.size(-1))
        pred_cam = self.init_cam.expand(B, -1)
        for i in range(n_iter) :
            xc = torch.cat([x, pred_cam], dim=1)    # [B, T, 256] 
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            pred_cam = self.deccam(xc) + pred_cam


"""Accumulated Token Module"""
class ATM(nn.Module):
    def __init__(self,
                 seqlen,
                 cam_encoder_depth,
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
        self.cam_output_proj = nn.Linear(embed_dim//2, 2048)
        self.cam_encoder = Transformer(depth=cam_encoder_depth, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*2, 
                                       h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                                       attn_drop_rate=attn_drop_rate, length=seqlen)

        ##########################
        # Accumulated Token
        ##########################
        self.context_tokenizer = CAM(seqlen=seqlen, d_model=2048, d_token=embed_dim)
        self.pose_shape_encoder = Transformer()
        self.fusing = CFM(embed_dim)


    def forward(self, x) :
        """
        x : [B, T, 2048]
        """
        ##########################
        # Camera parameter 
        ##########################
        cam_feat = self.cam_proj(x)                 # [B, T, 128]
        cam_feat = self.cam_encoder(cam_feat)       
        cam_feat = self.cam_output_proj()           # [B, T, 2048]

        ##########################
        # Accumulated Token
        ##########################
        context_feat = self.context_tokenizer(x)    # [B, T, 256]
        x = self.fusing(x, context_feat)
        ps_feat = self.pose_shape_encoder(x)
