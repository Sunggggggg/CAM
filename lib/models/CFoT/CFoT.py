import os
import torch
import torch.nn as nn

from lib.core.config import BASE_DATA_DIR
from lib.models.CFoT.transformer import Transformer
from lib.models.AccumulatedToken.enc_dec import ED_Transformer
from lib.models.AccumulatedToken.TSM import TSM
from lib.models.spin import Regressor
from lib.models.HSCR import HSCR

class CFoT(nn.Module):
    def __init__(self, 
                 seqlen=16,
                 cam_layer_depth=2,
                 po_sh_layer_depth=3,
                 embed_dim=512,
                 num_head=8,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 attn_drop_rate=0.0,
                 drop_reg_short=0.5,
                 pretrained=os.path.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
                 ) :
        super().__init__()
        self.seqlen = seqlen
        ##########################
        # Encoder
        ##########################
        self.input_proj = nn.Linear(2048, embed_dim)
        self.global_encoder = Transformer(depth=2, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*4, 
                    h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                    attn_drop_rate=attn_drop_rate, length=seqlen)
        
        self.global_decoder = Transformer(depth=1, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*4, 
                    h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                    attn_drop_rate=attn_drop_rate, length=seqlen)

        ##########################
        # Camera 
        ##########################
        cam_embed_dim = embed_dim // 4
        self.cam_proj = nn.Linear(2048, cam_embed_dim)
        self.cam_norm = nn.LayerNorm(cam_embed_dim)

        self.cam_dec = Transformer(depth=1, embed_dim=cam_embed_dim, mlp_hidden_dim=cam_embed_dim*4, 
                                       h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
                                       attn_drop_rate=attn_drop_rate, length=seqlen)

        ##########################
        # Pose, Shape 
        ##########################
        pose_shape_embed_dim = embed_dim // 2
        self.pose_shape_proj = nn.Linear(2048, pose_shape_embed_dim)
        self.pose_shape_norm = nn.LayerNorm(pose_shape_embed_dim)
        self.global_output = nn.Linear(embed_dim, 2048)
        self.local_output = nn.Linear(pose_shape_embed_dim + cam_embed_dim, 256)
        self.pose_shape_enc = Transformer(depth=po_sh_layer_depth, embed_dim=pose_shape_embed_dim,
                mlp_hidden_dim=pose_shape_embed_dim*4, h=num_head, drop_rate=drop_rate,
                drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, length=seqlen//2)
        
        self.local_regressor = HSCR(drop=drop_reg_short)
        self.global_regressor = Regressor()
        self.tsm = TSM(pose_shape_embed_dim)

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.global_regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    def split_frame(self, frame):
        even_frame = frame[:, ::2]
        odd_frame = frame[:, 1::2]
        return even_frame, odd_frame

    def cat_frame(self, even_frame, odd_frame) :
        frame = []
        for i in range(self.seqlen//2) :
            frame += [even_frame[:, i], odd_frame[:, i]]
        frame = torch.stack(frame, dim=1)
        return frame

    def forward(self, x, is_train=False, J_regressor=None):
        """
        x : [B, T, 2048]
        """
        B = x.shape[0]
        ##########################
        # Encoder
        ##########################
        x = self.input_proj(x)          # [B, T, 512]
        x = self.global_encoder(x)      # [B, T, 512]
        global_feat = self.global_decoder(x)

        if is_train :
            size = self.seqlen
            global_feat = self.global_output(global_feat)
        else :
            size = 1
            global_feat = self.global_output(global_feat)[:, self.seqlen // 2][:, None, :]
        
        smpl_output_global, pred_global = self.global_regressor(global_feat, is_train=is_train, J_regressor=J_regressor, n_iter=3)
        
        scores = None
        for s in smpl_output_global:
            s['theta'] = s['theta'].reshape(B, size, -1)           
            s['verts'] = s['verts'].reshape(B, size, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
            s['scores'] = scores

        ##########################
        # Camera 
        ##########################
        cam_feat = self.cam_proj(x)         # [B, T, 128]
        cam_feat = self.cam_norm(cam_feat)  # [B, T, 128]
        cam_feat = self.cam_dec(cam_feat)
        local_cam_feat = cam_feat[self.seqlen//2 - 1:self.seqlen//2 + 2]         

        ##########################
        # Pose, Shape 
        ##########################
        pose_shape_feat = self.pose_shape_proj(x)                   # [B, T, 256]
        pose_shape_feat = self.pose_shape_norm(pose_shape_feat)

        even_frame, odd_frame = self.split_frame(pose_shape_feat)   # even : past, odd : future
        pose_shape_feat = torch.cat([even_frame, odd_frame], dim=-1)
        local_pose_shape_feat = self.tsm(pose_shape_feat)                 # 

        local_feat = torch.cat([local_pose_shape_feat, local_cam_feat], dim=-1)    # [B, T, 256+128]
        local_feat = self.local_output(local_feat)
        
        if is_train:
            local_feat = local_feat
        else:
            local_feat = local_feat[:, 1][:, None, :]                           # [B, 1, 256]

        smpl_output = self.local_regressor(pose_shape_feat, init_pose=pred_global[0], init_shape=pred_global[1], init_cam=pred_global[2], is_train=is_train, J_regressor=J_regressor)
        
        if not is_train:    # Eval
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)         
                s['verts'] = s['verts'].reshape(B, -1, 3)      
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)
                s['scores'] = scores

        else:
            size = 3
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)           # [B, 3, 10]
                s['verts'] = s['verts'].reshape(B, size, -1, 3)        # [B, 3, 6980]
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)        # [B, 3, 2]
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)        # [B, 3, 3]
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)   # [B, 3, 3, 3]
                s['scores'] = scores

        return smpl_output, smpl_output_global