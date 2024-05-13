import numpy as np
import torch
import torch.nn as nn

from lib.models.trans_operator import Mlp
from lib.models.transformer import Transformer
from lib.models.smpl import SMPL_MEAN_PARAMS, SMPL, SMPL_MODEL_DIR
from lib.models.HSCR import KTD
from lib.models.spin import projection
from lib.utils.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis

class Layer(nn.Module) :
    def __init__(self, 
                 in_seqlen, 
                 out_seqlen, 
                 in_dim, 
                 out_dim,
                 depth,
                 num_head=8,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 attn_drop=0.,
                 stream="DOWN"
                 ) :
        super().__init__()
        self.stream = stream
        if stream == "DOWN":
            self.transformer = Transformer(depth=depth, embed_dim=in_dim, mlp_hidden_dim=in_dim*2,
                h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop, length=in_seqlen)
        else :
            self.transformer = Transformer(depth=depth, embed_dim=out_dim, mlp_hidden_dim=out_dim*2,
                h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop, length=out_seqlen)
        
        self.sampling = nn.Linear(in_seqlen, out_seqlen)
        self.c_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x) :
        """
        x : [B, T, C]
        """
        if self.stream == "DOWN":
            x = self.transformer(x)     # [B, T, C]
            x = x.permute(0, 2, 1)      # [B, C, T]
            x = self.sampling(x)        # [B, C, T/2]
            x = x.permute(0, 2, 1)      # [B, T/2, C]
            x = self.c_proj(x)          # [B, T/2, 2C]
            x = self.norm(x)            # [B, T/2, 2C]
            x = self.relu(x)            # [B, T/2, 2C]
        else :
            x = x.permute(0, 2, 1)      # [B, C, T]
            x = self.sampling(x)        # [B, C, 2T]
            x = x.permute(0, 2, 1)      # [B, 2T, C]
            x = self.c_proj(x)          # [B, 2T, C/2]
            x = self.norm(x)            # [B, 2T, C/2]
            x = self.relu(x)            # 
            x = self.transformer(x)     # [B, 2T, C/2]

        return x

class Regressor(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, hidden_dim=1024, drop=0.5) :
        super().__init__()
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)   # [1, 114]
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0) # [1, 10]
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0) # [1, 3]
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        npose = 24 * 6
        self.fc1 = nn.Linear(256 + 10, hidden_dim)
        self.fc2 = nn.Linear(256 + npose, hidden_dim)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

        self.decshape = nn.Linear(hidden_dim, 10)
        self.deccam = nn.Linear(hidden_dim * 2 + 3, 3)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False,
        )

        self.local_reg = KTD(hidden_dim)

    def forward(self, x) :
        """
        x : [B, T, 256] mid frame
        """
        B, T = x.shape[:2]
        init_pose = self.init_pose.expand(B, T, -1)
        init_shape = self.init_shape.expand(B, T, -1)
        init_cam = self.init_cam.expand(B, T, -1)

        xc_shape_cam = torch.cat([x, init_shape], -1)
        xc_pose_cam = torch.cat([x, init_pose], -1)

        xc_shape_cam = self.fc1(xc_shape_cam)           # [B, 1, 256+10] => [B, 1, hidden_dim]
        xc_shape_cam = self.drop1(xc_shape_cam)

        xc_pose_cam = self.fc2(xc_pose_cam)             # [B, 1, 256+144] => [B, 1, hidden_dim]
        xc_pose_cam = self.drop2(xc_pose_cam)
       
        pred_pose = self.local_reg(xc_pose_cam, init_pose) + init_pose
        pred_shape = self.decshape(xc_shape_cam) + init_shape  
        pred_cam = self.deccam(torch.cat([xc_pose_cam, xc_shape_cam, init_cam], -1)) + init_cam

        pred_pose = pred_pose.reshape(-1, 144)      # [B, 24*6]
        pred_shape = pred_shape.reshape(-1, 10)     # [B, 10]
        pred_cam = pred_cam.reshape(-1, 3)          # [B, 3]
        batch_size = pred_pose.shape[0]

        out_put = self.get_output(pred_pose, pred_shape, pred_cam, batch_size)
        return out_put

    def get_output(self, pred_pose, pred_shape, pred_cam, batch_size):
        """
        pred_pose   : [B, 24*6]
        pred_shape  : [B, 10]
        pred_cam    : [B, 3]
        """
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3) # [B, 24, 3, 3]
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )

        pred_vertices = pred_output.vertices        # [B, 6890, 3]
        pred_joints = pred_output.joints            # [B, 49, 3]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]
        return output
        
"UNet Modeling for humanMeshRecostruction"
class UMR(nn.Module):
    def __init__(self,
                 seqlen=16,
                 d_model=256,
                 num_head=8,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 attn_drop=0.
                 ):
        super().__init__()
        self.seqlen = seqlen
        self.input_proj = nn.Linear(2048, d_model)
        self.output_proj = nn.Linear(d_model, 2048)
        
        self.down1 = Layer(in_seqlen=seqlen, out_seqlen=seqlen//2, in_dim=d_model, out_dim=d_model*2, depth=1,
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="DOWN")

        self.down2 = Layer(in_seqlen=seqlen//2, out_seqlen=seqlen//4, in_dim=d_model*2, out_dim=d_model*4, depth=2,
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="DOWN")
        
        self.down3 = Layer(in_seqlen=seqlen//4, out_seqlen=seqlen//8, in_dim=d_model*4, out_dim=d_model*8, depth=3,
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="DOWN")
        
        self.up3 = Layer(in_seqlen=seqlen//8, out_seqlen=seqlen//4, in_dim=d_model*8, out_dim=d_model*4, depth=3, 
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="UP")
        
        self.up2 = Layer(in_seqlen=seqlen//4, out_seqlen=seqlen//2, in_dim=d_model*4, out_dim=d_model*2, depth=2,
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="UP")
        
        self.up1 = Layer(in_seqlen=seqlen//2, out_seqlen=seqlen, in_dim=d_model*2, out_dim=d_model, depth=1,
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="UP")
        
        #self.mlp3 = Mlp(in_features=d_model*8, out_features=d_model*4)
        #self.mlp2 = Mlp(in_features=d_model*4, out_features=d_model*2)
        #self.mlp1 = Mlp(in_features=d_model*2, out_features=d_model)

        # Regressor
        self.regressor = Regressor()

    def forward(self, x, is_train=False) :
        """
        x : [B, T, C]
        """
        x = self.input_proj(x)  # [B, T, 256]

        x1 = self.down1(x)      # [B, T, 256]   -> [B, T/2, 512]
        x2 = self.down2(x1)     # [B, T/2, 512] -> [B, T/4, 1024]
        x3 = self.down3(x2)     # [B, T/4, 1024]-> [B, T/8, 2048]

        x4 = self.up3(x3)   # [B, T/8, 2048] -> [B, T/4, 1024]
        x5 = self.up2(x4)   # [B, T/4, 1024] -> [B, T/2, 512]
        x6 = self.up1(x5)   # [B, T/2, 512] ->  [B, T, 256]
        
        if is_train :
            x_out = x6
        else :
            x_out = x6[:, self.seqlen//2:self.seqlen//2+1]  # [B, 1, 256]
            
        smpl_output = self.regressor(x_out)                 # 

        return smpl_output

if __name__ == "__main__":
    x = torch.randn((1, 16, 2048))
    model = UMR()
    model(x)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)) 