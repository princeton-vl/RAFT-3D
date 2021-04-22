import sys
sys.path.append('.')

from tqdm import tqdm
import os
import numpy as np
import cv2
import argparse
import torch

from lietorch import SE3
import raft3d.projective_ops as pops

from utils import show_image, normalize_image
from data_readers.kitti import KITTIEval
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from glob import glob
from data_readers.frame_utils import *


def display(img, tau, phi):
    """ display se3 fields """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(img[:, :, ::-1] / 255.0)

    tau_img = np.clip(tau, -0.1, 0.1)
    tau_img = (tau_img + 0.1) / 0.2

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    ax2.imshow(tau_img)
    ax3.imshow(phi_img)
    plt.show()


def prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=1.0):
    """ padding, normalization, and scaling """
    
    ht, wd = image1.shape[-2:]
    pad_h = (-ht) % 8
    pad_w = (-wd) % 8

    image1 = F.pad(image1, [0,pad_w,0,pad_h], mode='replicate')
    image2 = F.pad(image2, [0,pad_w,0,pad_h], mode='replicate')
    depth1 = F.pad(depth1[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]
    depth2 = F.pad(depth2[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]

    depth1 = (depth_scale * depth1).float()
    depth2 = (depth_scale * depth2).float()
    image1 = normalize_image(image1.float())
    image2 = normalize_image(image2.float())

    depth1 = depth1.float()
    depth2 = depth2.float()

    return image1, image2, depth1, depth2, (pad_w, pad_h)


@torch.no_grad()
def make_kitti_submission(model):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 1, 'drop_last': False}
    test_loader = DataLoader(KITTIEval(), **loader_args)

    DEPTH_SCALE = .1

    for i_batch, data_blob in enumerate(test_loader):
        image1, image2, disp1, disp2, intrinsics = [item.cuda() for item in data_blob]

        img1 = image1[0].permute(1,2,0).cpu().numpy()
        depth1 = DEPTH_SCALE * (intrinsics[0,0] / disp1)
        depth2 = DEPTH_SCALE * (intrinsics[0,0] / disp2)

        ht, wd = image1.shape[2:]
        image1, image2, depth1, depth2, _ = \
            prepare_images_and_depths(image1, image2, depth1, depth2)

        Ts = model(image1, image2, depth1, depth2, intrinsics, iters=16)
        tau_phi = Ts.log()

        # uncomment to diplay motion field
        # tau, phi = Ts.log().split([3,3], dim=-1)
        # tau = tau[0].cpu().numpy()
        # phi = phi[0].cpu().numpy()
        # display(img1, tau, phi)

        # compute optical flow
        flow, _, _ = pops.induced_flow(Ts, depth1, intrinsics)
        flow = flow[0, :ht, :wd, :2].cpu().numpy()

        # compute disparity change
        coords, _ = pops.projective_transform(Ts, depth1, intrinsics)
        disp2 =  intrinsics[0,0] * coords[:,:ht,:wd,2] * DEPTH_SCALE
        disp1 = disp1[0].cpu().numpy()
        disp2 = disp2[0].cpu().numpy()

        KITTIEval.write_prediction(i_batch, disp1, disp2, flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path the model weights')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    parser.add_argument('--radius', type=int, default=32)
    args = parser.parse_args()

    import importlib
    RAFT3D = importlib.import_module(args.network).RAFT3D

    model = torch.nn.DataParallel(RAFT3D(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    if not os.path.isdir('kitti_submission'):
        os.mkdir('kitti_submission')
        os.mkdir('kitti_submission/disp_0')
        os.mkdir('kitti_submission/disp_1')
        os.mkdir('kitti_submission/flow')

    make_kitti_submission(model)
