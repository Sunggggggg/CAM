# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import sys


sys.path.append('.')

import os
import cv2
import torch
import joblib
import argparse
import numpy as np
import pickle as pkl
import os.path as osp
from tqdm import tqdm

from lib.models import spin
from lib.data_utils._kp_utils import *
from lib.core.config import GLoT_DB_DIR, BASE_DATA_DIR
from lib.utils.smooth_bbox import get_smooth_bbox_params, get_all_bbox_params
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.data_utils._feature_extractor import extract_features, appear_extract_features
from lib.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis
from lib.data_utils._occ_utils import load_occluders
from lib.utils.vis import draw_skeleton
from lib.models.appearance import appear_extractor

# #####
# from ViTPose.mmpose.apis import inference_top_down_pose_model, init_pose_model
# from ViTPose.mmpose.datasets import DatasetInfo

NUM_JOINTS = 24
VIS_THRESH = 0.3
MIN_KP = 6

def read_data(folder, set, debug=False):
    dataset = {
        'appear_features': []
    }
    # model = spin.get_pretrained_hmr()
    model = appear_extractor()

    sequences = [x.split('.')[0] for x in os.listdir(osp.join(folder, 'sequenceFiles', set))]

    J_regressor = None

    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    if set == 'test' or set == 'validation':
        J_regressor = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    for i, seq in tqdm(enumerate(sequences)):

        data_file = osp.join(folder, 'sequenceFiles', set, seq + '.pkl')

        data = pkl.load(open(data_file, 'rb'), encoding='latin1')

        img_dir = osp.join(folder, 'imageFiles', seq)

        num_people = len(data['poses'])
        num_frames = len(data['img_frame_ids'])
        assert (data['poses2d'][0].shape[0] == num_frames)

        for p_id in range(num_people):
            j2d = data['poses2d'][p_id].transpose(0,2,1)

            img_paths = []
            for i_frame in range(num_frames):
                img_path = os.path.join(img_dir + '/image_{:05d}.jpg'.format(i_frame))
                img_paths.append(img_path)

            bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2d, vis_thresh=VIS_THRESH, sigma=8)

            img_paths_array = np.array(img_paths)[time_pt1:time_pt2]
            ############################
            appear_features = appear_extract_features(model, img_paths_array)
            dataset['appear_features'].append(appear_features)

            torch.cuda.empty_cache()

    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])
        print(k, dataset[k].shape)

    # Comment this code to get all frames for rendering
    # Filter out keypoints
    indices_to_use = np.where((dataset['joints2D'][:, :, 2] > VIS_THRESH).sum(-1) > MIN_KP)[0]
    for k in dataset.keys():
        dataset[k] = dataset[k][indices_to_use]

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/3dpw')
    args = parser.parse_args()

    debug = False
    dir = '/mnt/SKY/data/preprocessed_data/FullFrame_layerbranch/'
    dataset = read_data(args.dir, 'validation', debug=debug)
    joblib.dump(dataset, osp.join(dir, '3dpw_val_appear_db.pt'))

    dataset = read_data(args.dir, 'test', debug=debug)
    joblib.dump(dataset, osp.join(dir, '3dpw_test_appear_db.pt'))

    dataset = read_data(args.dir, 'train', debug=debug)
    joblib.dump(dataset, osp.join(dir, '3dpw_train_appear_db.pt'))
