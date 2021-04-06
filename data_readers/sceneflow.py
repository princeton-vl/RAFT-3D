
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import cv2
import math
import random
import json
import pickle
import os.path as osp

from glob import glob

import raft3d.projective_ops as pops
from .augmentation import RGBDAugmentor, SparseAugmentor
from . import frame_utils


def convert_blender_poses(poses):
    blender_to_world = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]])

    blender_to_world = blender_to_world.astype(np.float32)
    return np.matmul(blender_to_world, poses)

class FlyingThingsTest(data.Dataset):
    def __init__(self, root='datasets/FlyingThings3D'):

        self.dataset_index = []
        test_data = pickle.load(open('datasets/things_test_data.pickle', 'rb'))

        for (data_paths, sampled_pix1_x, sampled_pix2_y, mask) in test_data:
            split, subset, sequence, camera, frame = data_paths.split('_')
            sampled_pix1_x = sampled_pix1_x[mask]
            sampled_pix2_y = 539 - sampled_pix2_y[mask]
            sampled_index = np.stack([sampled_pix2_y, sampled_pix1_x], axis=0)

            # intrinsics
            fx, fy, cx, cy = (1050.0, 1050.0, 480.0, 270.0)
            intrinsics = np.array([fx, fy, cx, cy])

            frame = int(frame)
            image1 = osp.join(root, 'frames_cleanpass', split, subset, sequence, camera, "%04d.png" % (frame))
            image2 = osp.join(root, 'frames_cleanpass', split, subset, sequence, camera, "%04d.png" % (frame + 1))

            disp1 = osp.join(root, 'disparity', split, subset, sequence, camera, "%04d.pfm" % (frame))
            disp2 = osp.join(root, 'disparity', split, subset, sequence, camera, "%04d.pfm" % (frame + 1))

            if camera == 'left':
                flow = osp.join(root, 'optical_flow', split, subset, sequence, 
                    'into_future', camera, "OpticalFlowIntoFuture_%04d_L.pfm" % (frame))
                disparity_change = osp.join(root, 'disparity_change', split, subset, 
                    sequence, 'into_future', camera, "%04d.pfm" % (frame))
            
            else:
                flow = osp.join(root, 'optical_flow', split, subset, sequence, 
                    'into_future', camera, "OpticalFlowIntoFuture_%04d_R.pfm" % (frame))
                disparity_change = osp.join(root, 'disparity_change', split, subset, 
                    sequence, 'into_future', camera, "%04d.pfm" % (frame))

            datum = (image1, image2, disp1, disp2, flow, disparity_change, intrinsics, sampled_index)
            self.dataset_index.append(datum)


    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index):
        
        image1, image2, disp1, disp2, flow, disparity_change, intrinsics, sampled_index = self.dataset_index[index]
        
        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
        image1 = torch.from_numpy(image1).permute(2,0,1).float()
        image2 = torch.from_numpy(image2).permute(2,0,1).float()

        disp1 = frame_utils.read_gen(disp1)
        disp2 = frame_utils.read_gen(disp2)

        flow2d = frame_utils.read_gen(flow)[..., :2]
        disparity_change = frame_utils.read_gen(disparity_change)

        depth1 = torch.from_numpy(intrinsics[0] / disp1).float()
        depth2 = torch.from_numpy(intrinsics[0] / disp2).float()

        # transformed depth
        depth12 = torch.from_numpy(intrinsics[0] / (disp1 + disparity_change)).float()

        sampled_index = torch.from_numpy(sampled_index)
        intrinsics = torch.from_numpy(intrinsics).float()
        flow3d = pops.backproject_flow3d(flow2d, depth1, depth12, intrinsics)

        return image1, image2, depth1, depth2, flow2d, flow3d, intrinsics, sampled_index