import torch
import torch.nn as nn
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from lib.models.spin import projection

class Regressor(nn.Module):
    def __init__(self, d_model, smpl_mean_params=SMPL_MEAN_PARAMS):
        super(Regressor, self).__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(d_model + npose + 13, 1024)    #  2048 + 24*6(pose) + 10(shape) + 3(cam)
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, init_pose, init_shape, init_cam, n_iter=3, is_train=False, J_regressor=None):
        seq_len = x.shape[1]
        x = x.reshape(-1, x.size(-1))
        init_pose = init_pose.reshape(-1, init_pose.size(-1))
        init_shape = init_shape.reshape(-1, init_shape.size(-1))
        init_cam = init_cam.reshape(-1, init_cam.size(-1))

        batch_size = x.shape[0]

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