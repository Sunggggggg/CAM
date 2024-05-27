import os
import torch
import torch.nn as nn

from lib.models.OnlyConv.regressor import Regressor
from lib.core.config import BASE_DATA_DIR
from lib.models.HSCR import HSCR

class OC(nn.Module):
    def __init__(self,
                 seqlen=16,
                 pretrained=os.path.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),) :
        super().__init__()
        self.seqlen = seqlen
        self.conv1 = nn.Sequential(
            nn.Linear(seqlen, seqlen),
            nn.LayerNorm(seqlen),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(seqlen//2, seqlen//2),
            nn.LayerNorm(seqlen//2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(seqlen//4, seqlen//4),
            nn.LayerNorm(seqlen//4),
            nn.ReLU()
        )

        self.regressor_local = HSCR()
        self.regressor = Regressor()
        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')
 
    def forward(self, x, is_train=False, J_regressor=None):
        B = x.shape[0]
        init_x = x
        x = x.permute(0, 2, 1)
        x = self.conv1(x)                               # [B, 2048, 16]
        x = x[..., self.seqlen//2-4 : self.seqlen//2+4] # [B, 2048, 8]
        x = self.conv2(x)                               # [B, 2048, 8]
        x = x[..., self.seqlen//4-2 : self.seqlen//4+2] # [B, 2048, 4]
        x = self.conv3(x)
        x = x[..., self.seqlen//8-1: self.seqlen//8+1]    # [B, ]
        
        x = x.permute(0, 2, 1)  
        x = torch.mean(x, dim=1, keepdim=True)

        smpl_output_global, pred_global = self.regressor(init_x, is_train=is_train, J_regressor=J_regressor, n_iter=3)
        smpl_output = self.regressor(x, init_pose=pred_global[0], init_shape=pred_global[1], init_cam=pred_global[2], is_train=is_train, J_regressor=J_regressor)
        
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
        
        return smpl_output, smpl_output_global