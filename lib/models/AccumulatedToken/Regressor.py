import numpy as np
import torch
import torch.nn as nn
from lib.models.smpl import SMPL_MEAN_PARAMS

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
class Regressor(nn.Module):
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

from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
smpl = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)
def regressor_output(pred_pose, pred_shape, pred_cam, J_regressor=None) :
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
