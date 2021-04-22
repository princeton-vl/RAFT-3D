
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import cv2
import math
import random
import json
import csv
import pickle
import os.path as osp

from glob import glob

import raft3d.projective_ops as pops
from . import frame_utils
from .augmentation import RGBDAugmentor, SparseAugmentor

class KITTIEval(data.Dataset):

    crop = 80

    def __init__(self, image_size=None, root='datasets/KITTI', do_augment=True):
        self.init_seed = None
        mode = "testing"
        self.image1_list = sorted(glob(osp.join(root, mode, "image_2/*10.png")))
        self.image2_list = sorted(glob(osp.join(root, mode, "image_2/*11.png")))
        self.disp1_ga_list = sorted(glob(osp.join(root, mode, "disp_ganet_{}/*10.png".format(mode))))
        self.disp2_ga_list = sorted(glob(osp.join(root, mode, "disp_ganet_{}/*11.png".format(mode))))
        self.calib_list = sorted(glob(osp.join(root, mode, "calib_cam_to_cam/*.txt")))

        self.intrinsics_list = []
        for calib_file in self.calib_list:
            with open(calib_file) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[0] == 'K_02:':
                        K = np.array(row[1:], dtype=np.float32).reshape(3,3)
                        kvec = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
                        self.intrinsics_list.append(kvec)

    @staticmethod
    def write_prediction(index, disp1, disp2, flow):

        def writeFlowKITTI(filename, uv):
            uv = 64.0 * uv + 2**15
            valid = np.ones([uv.shape[0], uv.shape[1], 1])
            uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
            cv2.imwrite(filename, uv[..., ::-1])

        def writeDispKITTI(filename, disp):
            disp = (256 * disp).astype(np.uint16)
            cv2.imwrite(filename, disp)

        disp1 = np.pad(disp1, ((KITTIEval.crop,0),(0,0)), mode='edge')
        disp2 = np.pad(disp2, ((KITTIEval.crop, 0), (0,0)), mode='edge')
        flow = np.pad(flow, ((KITTIEval.crop, 0), (0,0),(0,0)), mode='edge')

        disp1_path = 'kitti_submission/disp_0/%06d_10.png' % index
        disp2_path = 'kitti_submission/disp_1/%06d_10.png' % index
        flow_path = 'kitti_submission/flow/%06d_10.png' % index

        writeDispKITTI(disp1_path, disp1)
        writeDispKITTI(disp2_path, disp2)
        writeFlowKITTI(flow_path, flow)
                        
    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):

        intrinsics = self.intrinsics_list[index]
        image1 = cv2.imread(self.image1_list[index])
        image2 = cv2.imread(self.image2_list[index])

        disp1 = cv2.imread(self.disp1_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2 = cv2.imread(self.disp2_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0

        image1 = image1[self.crop:]
        image2 = image2[self.crop:]
        disp1 = disp1[self.crop:]
        disp2 = disp2[self.crop:]
        intrinsics[3] -= self.crop

        image1 = torch.from_numpy(image1).float().permute(2,0,1)
        image2 = torch.from_numpy(image2).float().permute(2,0,1)
        disp1 = torch.from_numpy(disp1).float()
        disp2 = torch.from_numpy(disp2).float()
        intrinsics = torch.from_numpy(intrinsics).float()

        return image1, image2, disp1, disp2, intrinsics

