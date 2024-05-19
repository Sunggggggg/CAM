import numpy as np
import torch
import torch.nn as nn

from lib.models.HSCR import KTD
from lib.models.smpl import SMPL, SMPL_MEAN_PARAMS, H36M_TO_J14, SMPL_MODEL_DIR
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat

class Regressor(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS):
        super(Regressor, self).__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(512 * 4 + npose + 13, 1024)    #  2048 + 24*6(pose) + 10(shape) + 3(cam)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False,
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, is_train=False, J_regressor=None):
        seq_len = x.shape[1]
        x = x.reshape(-1, x.size(-1))
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if not is_train and J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            n_joint = 14
        else :
            n_joint = 49

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1).view(-1, seq_len, 85),
            'verts'  : pred_vertices.view(-1, seq_len, 6890, 3),
            'kp_2d'  : pred_keypoints_2d.view(-1, seq_len, n_joint, 2),
            'kp_3d'  : pred_joints.view(-1, seq_len, n_joint, 3),
            'rotmat' : pred_rotmat.view(-1, seq_len, 24, 3, 3)
        }]
        
        return output

class CamRegressor(nn.Module) :
    def __init__(self, d_model=256, smpl_mean_params=SMPL_MEAN_PARAMS) :
        super().__init__()
        mean_params = np.load(smpl_mean_params)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_cam', init_cam)  # 3

        self.fc1 = nn.Linear(3+d_model, d_model)
        self.drop1 = nn.Dropout()
        self.deccam = nn.Linear(d_model, 3)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

    def forward(self, x, n_iter=3) :
        """
        Input
            x : [B, T, 256]

        Return 
            pred_cam : [B, T, 3]
        """
        x = x.reshape(-1, x.size(-1))               # [BT, 256]
        BT = x.shape[0]
        pred_cam = self.init_cam.expand(BT, -1)     # [BT, 3]
        for i in range(n_iter) :
            xc = torch.cat([x, pred_cam], dim=-1)    # [BT, 256 + 3] 
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            pred_cam = self.deccam(xc) + pred_cam   # []

        return pred_cam
    
""" Pose, Shape Regressor"""
class PoseShapeRegressor(nn.Module):
    def __init__(self, d_model=256, smpl_mean_params=SMPL_MEAN_PARAMS) :
        super().__init__()
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)

        npose = 24 * 6

        self.fc1 = nn.Linear(d_model + npose + 10, 512)    #  2048 + 24*6(pose) + 10(shape) + 3(cam)
        self.drop1 = nn.Dropout()
        self.decpose = nn.Linear(512, npose)
        self.decshape = nn.Linear(512, 10)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)

    def forward(self, x, n_iter=3):
        """
        Input
            x : [B, T, 256]

        Return 
            pred_pose   : [B, T, 144]
            pred_shape  : [B, T, 10]
        """
        x = x.reshape(-1, x.size(-1))               # [BT, 256]
        BT = x.shape[0]
        pred_pose = self.init_pose.expand(BT, -1)       # [BT, 3]
        pred_shape = self.init_shape.expand(BT, -1)     # [BT, 3]

        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape], dim=-1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape

        return pred_pose, pred_shape

""" Total Regressor"""
class Total_Regressor(nn.Module):
    def __init__(self, d_model=256, smpl_mean_params=SMPL_MEAN_PARAMS, hidden_dim=1024, drop=0.5) :
        super().__init__()
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)

        self.register_buffer('init_cam', init_cam)  # 3
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)

        npose = 24 * 6

        self.proj = nn.Linear(d_model + npose + 10 + 3, hidden_dim)
        self.drop = nn.Dropout()

        self.decpose0 = nn.Linear(hidden_dim, npose)
        self.decshape0 = nn.Linear(hidden_dim, 10)
        self.deccam0 = nn.Linear(hidden_dim, 3)

        self.fc1 = nn.Linear(d_model + 10, hidden_dim)    #  2048 + 24*6(pose) + 10(shape) + 3(cam)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(d_model + npose, hidden_dim)
        self.drop2 = nn.Dropout(drop)

        self.decpose = nn.Linear(hidden_dim, npose)
        self.decshape = nn.Linear(hidden_dim, 10)
        self.deccam = nn.Linear(hidden_dim * 2 + 3, 3)

        self.local_reg = KTD(hidden_dim)
        
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

    def forward(self, x, n_iter=3):
        """
        Input
            x : [B, T, d]

        Return 
        """
        x = x.reshape(-1, x.size(-1))               # [BT, 256]
        BT = x.shape[0]

        pred_pose = self.init_pose.expand(BT, -1)       # [BT, 144]
        pred_shape = self.init_shape.expand(BT, -1)     # [BT, 10]
        pred_cam = self.init_cam.expand(BT, -1)         # [BT, 3]
        
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], dim=-1)
            xc = self.proj(xc)
            xc = self.drop(xc)

            pred_pose = self.decpose0(xc) + pred_pose
            pred_shape = self.decshape0(xc) + pred_shape
            pred_cam = self.deccam0(xc) + pred_cam

        xc_shape_cam = torch.cat([x, pred_shape], -1)   # [BT, 10+d]
        xc_pose_cam = torch.cat([x, pred_pose], -1)     # [BT, 144+d]

        xc_shape_cam = self.fc1(xc_shape_cam)           # [B, 1, 256+10] => [B, 1, hidden_dim]
        xc_shape_cam = self.drop1(xc_shape_cam)

        xc_pose_cam = self.fc2(xc_pose_cam)             # [B, 1, 256+144] => [B, 1, hidden_dim]
        xc_pose_cam = self.drop2(xc_pose_cam)
    
        pred_pose = self.local_reg(xc_pose_cam, pred_pose) + pred_pose
        pred_shape = self.decshape(xc_shape_cam) + pred_shape  
        pred_cam = self.deccam(torch.cat([xc_pose_cam, xc_shape_cam, pred_cam], -1)) + pred_cam

        pred_pose = pred_pose.reshape(-1, 144)      # [B, 24*6]
        pred_shape = pred_shape.reshape(-1, 10)     # [B, 10]
        pred_cam = pred_cam.reshape(-1, 3)          # [B, 3]

        return pred_pose, pred_shape, pred_cam

def regressor_output(smpl, pred_pose, pred_shape, pred_cam, seqlen, J_regressor=None) :
    """
    pred_pose, pred_shape, pred_cam : [BT, (24*6)(10)(3)]
    """
    BT = pred_cam.shape[0]
    pred_rotmat = rot6d_to_rotmat(pred_pose).view(BT, 24, 3, 3)

    pred_output = smpl(
        betas=pred_shape,
        body_pose=pred_rotmat[:, 1:],
        global_orient=pred_rotmat[:, 0].unsqueeze(1),
        pose2rot=False,
    )

    pred_vertices = pred_output.vertices
    pred_joints = pred_output.joints

    if J_regressor is not None:
        J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
        pred_joints = pred_joints[:, H36M_TO_J14, :]
        n_joint = 14
    else :
        n_joint = 49

    pred_keypoints_2d = projection(pred_joints, pred_cam)

    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

    output = [{
        'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1).view(-1, seqlen, 85),
        'verts'  : pred_vertices.view(-1, seqlen, 6890, 3),
        'kp_2d'  : pred_keypoints_2d.view(-1, seqlen, n_joint, 2),
        'kp_3d'  : pred_joints.view(-1, seqlen, n_joint, 3),
        'rotmat' : pred_rotmat.view(-1, seqlen, 24, 3, 3)
    }]

    return output

def projection(pred_joints, pred_camera):

    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

