import torch
import torch.nn.functional as F

from .sampler_ops import *

MIN_DEPTH = 0.05

def project(Xs, intrinsics):
    """ Pinhole camera projection """
    X, Y, Z = Xs.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[:,None,None].unbind(dim=-1)

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    d = 1.0 / Z

    coords = torch.stack([x, y, d], dim=-1)
    return coords

def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape[-2:]
    
    fx, fy, cx, cy = \
        intrinsics[:,None,None].unbind(dim=-1)

    y, x = torch.meshgrid(
        torch.arange(ht).to(depths.device).float(), 
        torch.arange(wd).to(depths.device).float())

    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths

    return torch.stack([X, Y, Z], dim=-1)

def projective_transform(Ts, depth, intrinsics):
    """ Project points from I1 to I2 """
    
    X0 = inv_project(depth, intrinsics)
    X1 = Ts * X0
    x1 = project(X1, intrinsics)

    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH)
    return x1, valid.float()

def induced_flow(Ts, depth, intrinsics):
    """ Compute 2d and 3d flow fields """

    X0 = inv_project(depth, intrinsics)
    X1 = Ts * X0

    x0 = project(X0, intrinsics)
    x1 = project(X1, intrinsics)

    flow2d = x1 - x0
    flow3d = X1 - X0

    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH)
    return flow2d, flow3d, valid.float()


def backproject_flow3d(flow2d, depth0, depth1, intrinsics):
    """ compute 3D flow from 2D flow + depth change """

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = \
        intrinsics[None].unbind(dim=-1)
    
    y0, x0 = torch.meshgrid(
        torch.arange(ht).to(depth0.device).float(), 
        torch.arange(wd).to(depth0.device).float())

    x1 = x0 + flow2d[...,0]
    y1 = y0 + flow2d[...,1]

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    X1 = depth1 * ((x1 - cx) / fx)
    Y1 = depth1 * ((y1 - cy) / fy)
    Z1 = depth1

    flow3d = torch.stack([X1-X0, Y1-Y0, Z1-Z0], dim=-1)
    return flow3d

