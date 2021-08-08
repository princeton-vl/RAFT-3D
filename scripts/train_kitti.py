import sys
sys.path.append('.')

import argparse
import cv2
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lietorch import SE3
import raft3d.projective_ops as pops

from utils import Logger, show_image, normalize_image
from evaluation import test_sceneflow

from data_readers.sceneflow import SceneFlow
from data_readers.kitti import KITTI


RV_WEIGHT = 0.2
DZ_WEIGHT = 250.0

def loss_fn(flow2d_est, flow2d_rev, flow_gt, valid_mask, gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    N = len(flow2d_est)
    loss = 0.0

    for i in range(N):
        w = gamma**(N - i - 1)
        fl_rev = flow2d_rev[i]

        fl_est, dz_est = flow2d_est[i].split([2,1], dim=-1)
        fl_gt, dz_gt = flow_gt.split([2,1], dim=-1)

        loss += w * (valid_mask * (fl_est - fl_gt).abs()).mean()
        loss += w * DZ_WEIGHT * (valid_mask * (dz_est - dz_gt).abs()).mean()
        loss += w * RV_WEIGHT * (valid_mask * (fl_rev - fl_gt).abs()).mean()

    epe_2d = (fl_est - fl_gt).norm(dim=-1)
    epe_2d = epe_2d.view(-1)[valid_mask.view(-1)]

    epe_dz = (dz_est - dz_gt).norm(dim=-1)
    epe_dz = epe_dz.view(-1)[valid_mask.view(-1)]

    metrics = {
        'epe2d': epe_2d.mean().item(),
        'epedz': epe_dz.mean().item(),
        '1px': (epe_2d < 1).float().mean().item(),
        '3px': (epe_2d < 3).float().mean().item(),
        '5px': (epe_2d < 5).float().mean().item(),
    }

    return loss, metrics


def fetch_dataloader(args):
    gpuargs = {'shuffle': True, 'num_workers': 4, 'drop_last' : True}
    train_dataset = KITTI(do_augment=True, image_size=[256, 960])    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **gpuargs)
    return train_loader


def fetch_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps, pct_start=0.001, cycle_momentum=False)
    return optimizer, scheduler


def train(args):

    import importlib
    RAFT3D = importlib.import_module(args.network).RAFT3D

    model = RAFT3D(args)
    model = torch.nn.DataParallel(model)

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    model.cuda()
    model.eval()
    
    logger = Logger()

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(model, args)

    total_steps = 0
    while 1:
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, depth1, depth2, flow_gt, valid, intrinsics = [x.cuda() for x in data_blob]

            image1 = normalize_image(image1.float())
            image2 = normalize_image(image2.float())

            flow2d_est, flow2d_rev = model(image1, image2, depth1, depth2, intrinsics, iters=12, train_mode=True)

            valid_mask = valid.unsqueeze(-1) > 0.5
            loss, metrics = loss_fn(flow2d_est, flow2d_rev, flow_gt, valid_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            logger.push(metrics)
            
            total_steps += 1

            if total_steps % 5000 == 0:
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--network', default='raft3d.raft3d', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--num_steps', type=int, default=50000)

    # model arguments
    parser.add_argument('--radius', type=int, default=32)
    
    args = parser.parse_args()

    print(args)
    train(args)

