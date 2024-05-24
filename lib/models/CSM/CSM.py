import os
import torch
import torch.nn as nn

from lib.models.CSM.transformer import ChannelTransformer
from lib.core.config import BASE_DATA_DIR
from lib.models.spin import Regressor
from lib.models.transformer import Transformer as local_transformer_encoder
from lib.models.trans_operator import CrossAttention
from lib.models.HSCR import HSCR

"""Channel Slicing Module"""
class CSM(nn.Module):
    def __init__(self,
                 seqlen=16,
                 d_model=2048,
                 slice=4,
                 num_head=8,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 attn_drop_rate=0.0,
                 drop_reg_short=0.5,
                 stride_short=4,
                 short_n_layers=2,
                 short_d_model=256,
                 pretrained=os.path.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
                 ) :
        super().__init__()
        self.seqlen = seqlen
        self.stride_short = stride_short
        self.mid_frame = seqlen // 2
        self.short_n_layers = short_n_layers

        enc_dim = d_model // slice
        self.channel_enc = ChannelTransformer(depth=3, embed_dim=enc_dim, mlp_hidden_dim=enc_dim*4,
                                              t_length=seqlen, s_length=slice, slice=slice, 
                                              h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
        
        dec_dim = enc_dim // 2
        self.dec_proj = nn.Linear(d_model, d_model//2)
        self.channel_dec = ChannelTransformer(depth=1, embed_dim=dec_dim, mlp_hidden_dim=dec_dim*4,
                                              t_length=seqlen, s_length=slice, slice=slice, 
                                              h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
        self.global_out_proj = nn.Linear(d_model//2, d_model)

        self.proj_short = nn.Linear(2048, short_d_model)
        self.proj_mem = nn.Linear(2048, short_d_model)
        self.local_trans_de = CrossAttention(short_d_model, num_heads=num_head, qkv_bias=False, \
        qk_scale=None, attn_drop=0., proj_drop=0.)
        self.local_trans_en = ChannelTransformer(depth=short_n_layers, embed_dim=short_d_model//slice, mlp_hidden_dim=short_d_model, 
                                                 t_length=self.stride_short * 2 + 1, s_length=slice, slice=slice, 
                                                 h=num_head, drop_rate=drop_rate, drop_path_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        self.regressor = HSCR(drop=drop_reg_short)

        self.global_regressor = Regressor()
        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.global_regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    def forward(self, x, is_train=False, J_regressor=None):
        B = x.shape[0]

        enc_x = self.channel_enc(x)                                # [B, T, 2048]
        global_feat = self.channel_dec(self.dec_proj(enc_x))       # [B, T, 512]

        if is_train :
            size = self.seqlen
            global_feat = self.global_out_proj(global_feat)
        else :
            size = 1
            global_feat = self.global_out_proj(global_feat)[:, self.seqlen // 2][:, None, :]

        smpl_output_global, pred_global = self.global_regressor(global_feat, is_train=is_train, J_regressor=J_regressor, n_iter=3)
        
        scores = None
        for s in smpl_output_global:
            s['theta'] = s['theta'].reshape(B, size, -1)           
            s['verts'] = s['verts'].reshape(B, size, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
            s['scores'] = scores

        x_short = x[:, self.mid_frame - self.stride_short:self.mid_frame + self.stride_short + 1]   # [B, 9, 2048]
        x_short = self.proj_short(x_short)                                  # [B, 9, 256]
        x_short = self.local_trans_en(x_short)                              # [B, L, 256] L=9 
        mid_fea = x_short[:, self.stride_short - 1: self.stride_short + 2]  # [B, 3, 256]
        mem = self.proj_mem(enc_x)                                          # [B, T/2, 256]
        out_short = self.local_trans_de(mid_fea, mem)                       # [B, 3, 256]
        
        if is_train:
            feature = out_short
        else:
            feature = out_short[:, 1][:, None, :]                           # [B, 1, 256]

        smpl_output = self.regressor(feature, init_pose=pred_global[0], init_shape=pred_global[1], init_cam=pred_global[2], is_train=is_train, J_regressor=J_regressor)
        
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

        
