
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


from scipy.spatial.transform import Rotation

def convert_blender_poses(poses):
    blender_to_world = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]])

    blender_to_world = blender_to_world.astype(np.float32)
    return np.matmul(blender_to_world, poses)

def read_camdata(cam_file):
    poses = []
    with open(cam_file) as f:
        for line in f:
            if 'L' == line[0]:
                pose = np.array(line[2:].split(' '))
                pose = pose.reshape(4, 4).astype(np.float)
                pose = np.linalg.inv(pose)
                pose = convert_blender_poses(pose)
                pose = np.linalg.inv(pose)

                R, t = pose[:3, :3], pose[:3, 3]
                q = Rotation.from_matrix(R).as_quat()
                poses.append(np.concatenate([t, q], 0))
    
    return np.stack(poses, 0)

class SceneFlow(data.Dataset):
    def __init__(self, image_size=None,
                 n_frames=2,
                 mode='TRAIN', 
                 do_augment=True, 
                 root='datasets',
                 dstype='frames_cleanpass', 
                 use_flyingthings=True, 
                 use_monkaa=False, 
                 use_driving=False):

        self.init_seed = None
        
        self.do_augment = do_augment
        self.n_frames = n_frames
        self.mode = mode
        self.root = root
        self.dstype = dstype

        self.image_list = []
        self.depth_list = []
        self.delta_list = []
        self.pose_list = []
        self.flow_list = []
        self.intrinsics_list = []

        if self.mode == 'TRAIN':

            if self.do_augment:
                self.augmentor = RGBDAugmentor(image_size)

            if use_flyingthings:
                self.add_flyingthings()
            print(len(self.image_list))
            
            if use_monkaa:
                self.add_monkaa()
            print(len(self.image_list))

            if use_driving:
                self.add_driving()
            print(len(self.image_list))

        elif self.mode == 'TEST':
            self.add_flyingthings()

    def read_camdata(self, cam_file):
        poses = []
        with open(cam_file) as f:
            for line in f:
                if 'L' == line[0]:
                    pose = np.array(line[2:].split(' '))
                    pose = pose.reshape(4, 4).astype(np.float)
                    poses.append(pose)
        return poses
                

    def add_flyingthings(self, mode='TRAIN'):
        root = osp.join(self.root, 'FlyingThings3D')

        exclude = np.loadtxt('misc/exclude.txt', delimiter=' ', dtype=np.unicode_)
        exclude = set(exclude)

        # intrinsics
        fx = 1050.0
        fy = 1050.0
        cx = 480.0
        cy = 270.0

        for cam in ['left', 'right']:
            image_dirs = sorted(glob(osp.join(root, self.dstype, self.mode, '*/*')))
            image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

            flow_dirs = sorted(glob(osp.join(root, 'optical_flow', self.mode, '*/*')))
            flow_dirs_forw = sorted([osp.join(f, 'into_future', cam) for f in flow_dirs])
            flow_dirs_back = sorted([osp.join(f, 'into_past', cam) for f in flow_dirs])

            depth_dirs = sorted(glob(osp.join(root, 'disparity', self.mode, '*/*')))
            depth_dirs = sorted([osp.join(f, cam) for f in depth_dirs])

            delta_dirs = sorted(glob(osp.join(root, 'disparity_change', self.mode, '*/*')))
            delta_dirs_forw = sorted([osp.join(f, 'into_future', cam) for f in delta_dirs])
            delta_dirs_back = sorted([osp.join(f, 'into_past', cam) for f in delta_dirs])

            label_dirs = depth_dirs
            cam_dirs = sorted(glob(osp.join(root, 'camera_data', self.mode, '*/*')))

            for idir, fdir_forw, fdir_back, ddir_forw, ddir_back, zdir, cdir, ldir in \
                    zip(image_dirs, flow_dirs_forw, flow_dirs_back, delta_dirs_forw, delta_dirs_back, depth_dirs, cam_dirs, label_dirs):
                
                images = sorted(glob(osp.join(idir, '*.png')))
                flows_forw = sorted(glob(osp.join(fdir_forw, '*.pfm')))
                flows_back = sorted(glob(osp.join(fdir_back, '*.pfm')))

                delta_forw = sorted(glob(osp.join(ddir_forw, '*.pfm')))
                delta_back = sorted(glob(osp.join(ddir_back, '*.pfm')))
                
                depths = sorted(glob(osp.join(zdir, '*.pfm')))
                labels = sorted(glob(osp.join(ldir, '*.pfm')))
                
                poses = read_camdata(osp.join(cdir, 'camera_data.txt'))
                if len(poses) < len(images):
                    continue

                for i in range(1, len(images)-1):
                    tag = '/'.join(images[i].split('/')[-5:])
                    if tag in exclude:
                        # print("Excluding %s" % tag)
                        continue

                    self.intrinsics_list += [np.array([fx, fy, cx, cy])]
                    self.image_list += [[images[i], images[i+1]]]
                    self.flow_list += [flows_forw[i]]
                    self.delta_list += [delta_forw[i]]
                    self.pose_list += [[poses[i], poses[i+1]]]
                    self.depth_list += [[depths[i], depths[i+1]]]

                    self.intrinsics_list += [np.array([fx, fy, cx, cy])]
                    self.image_list += [[images[i], images[i-1]]]
                    self.flow_list += [flows_back[i]]
                    self.delta_list += [delta_back[i]]
                    self.pose_list += [[poses[i], poses[i-1]]]
                    self.depth_list += [[depths[i], depths[i-1]]]

    
    def add_monkaa(self):
        root = osp.join(self.root, 'Monkaa')

        # intrinsics
        fx = 1050.0
        fy = 1050.0
        cx = 479.5
        cy = 269.5

        for cam in ['left']:
            image_dirs = sorted(glob(osp.join(root, self.dstype, '*')))
            image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

            flow_dirs = sorted(glob(osp.join(root, 'optical_flow/*')))
            flow_dirs_forw = sorted([osp.join(f, 'into_future', cam) for f in flow_dirs])
            flow_dirs_back = sorted([osp.join(f, 'into_past', cam) for f in flow_dirs])


            delta_dirs = sorted(glob(osp.join(root, 'disparity_change/*')))
            delta_dirs_forw = sorted([osp.join(f, 'into_future', cam) for f in delta_dirs])
            delta_dirs_back = sorted([osp.join(f, 'into_past', cam) for f in delta_dirs])

            depth_dirs = sorted(glob(osp.join(root, 'disparity/*')))
            depth_dirs = sorted([osp.join(f, cam) for f in depth_dirs])
            
            label_dirs = depth_dirs

            # label_dirs = sorted(glob(osp.join(root, 'object_index/*')))
            # label_dirs = sorted([osp.join(f, cam) for f in label_dirs])

            cam_dirs = sorted(glob(osp.join(root, 'camera_data/*')))
            for idir, fdir_forw, fdir_back, ddir_forw, ddir_back, zdir, cdir, ldir in \
                    zip(image_dirs, flow_dirs_forw, flow_dirs_back, delta_dirs_forw, delta_dirs_back, depth_dirs, cam_dirs, label_dirs):
                
                images = sorted(glob(osp.join(idir, '*.png')))
                flows_forw = sorted(glob(osp.join(fdir_forw, '*.pfm')))
                flows_back = sorted(glob(osp.join(fdir_back, '*.pfm')))

                delta_forw = sorted(glob(osp.join(ddir_forw, '*.pfm')))
                delta_back = sorted(glob(osp.join(ddir_back, '*.pfm')))
                
                depths = sorted(glob(osp.join(zdir, '*.pfm')))
                labels = sorted(glob(osp.join(ldir, '*.pfm')))
                
                poses = read_camdata(osp.join(cdir, 'camera_data.txt'))
                if len(poses) < len(images):
                    continue

                for i in range(1, len(images)-1):
                    self.intrinsics_list += [np.array([fx, fy, cx, cy])]
                    self.image_list += [[images[i], images[i+1]]]
                    self.flow_list += [flows_forw[i]]
                    self.delta_list += [delta_forw[i]]
                    self.pose_list += [[poses[i], poses[i+1]]]
                    self.depth_list += [[depths[i], depths[i+1]]]

                    self.intrinsics_list += [np.array([fx, fy, cx, cy])]
                    self.image_list += [[images[i], images[i-1]]]
                    self.flow_list += [flows_back[i]]
                    self.delta_list += [delta_back[i]]
                    self.pose_list += [[poses[i], poses[i-1]]]
                    self.depth_list += [[depths[i], depths[i-1]]]

    def add_driving(self):
        root = osp.join(self.root, 'Driving')

        # intrinsics
        cx = 480.0
        cy = 270.0

        for cam in ['left']:
            image_dirs = sorted(glob(osp.join(root, self.dstype, '*/*/*')))
            image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

            flow_dirs = sorted(glob(osp.join(root, 'optical_flow/*/*/*')))
            flow_dirs_forw = sorted([osp.join(f, 'into_future', cam) for f in flow_dirs])
            flow_dirs_back = sorted([osp.join(f, 'into_past', cam) for f in flow_dirs])

            delta_dirs = sorted(glob(osp.join(root, 'disparity_change/*/*/*')))
            delta_dirs_forw = sorted([osp.join(f, 'into_future', cam) for f in delta_dirs])
            delta_dirs_back = sorted([osp.join(f, 'into_past', cam) for f in delta_dirs])

            depth_dirs = sorted(glob(osp.join(root, 'disparity/*/*/*')))
            depth_dirs = sorted([osp.join(f, cam) for f in depth_dirs])

            label_dirs = sorted(glob(osp.join(root, 'object_index/*/*/*')))
            label_dirs = sorted([osp.join(f, cam) for f in label_dirs])

            cam_dirs = sorted(glob(osp.join(root, 'camera_data/*/*/*')))

            for idir, fdir_forw, fdir_back, ddir_forw, ddir_back, zdir, cdir, ldir in \
                    zip(image_dirs, flow_dirs_forw, flow_dirs_back, delta_dirs_forw, delta_dirs_back, depth_dirs, cam_dirs, label_dirs):

                if '15mm_focallength' in idir:
                    fx = fy = 450.0
                elif '35mm_focallength' in idir:
                    fx = fy = 1050.0
                
                images = sorted(glob(osp.join(idir, '*.png')))
                flows_forw = sorted(glob(osp.join(fdir_forw, '*.pfm')))
                flows_back = sorted(glob(osp.join(fdir_back, '*.pfm')))

                delta_forw = sorted(glob(osp.join(ddir_forw, '*.pfm')))
                delta_back = sorted(glob(osp.join(ddir_back, '*.pfm')))
                
                depths = sorted(glob(osp.join(zdir, '*.pfm')))
                labels = sorted(glob(osp.join(ldir, '*.pfm')))
                
                poses = self.read_camdata(osp.join(cdir, 'camera_data.txt'))

                if len(poses) < len(images):
                    continue

                for i in range(1, len(images)-1):
                    self.intrinsics_list += [np.array([fx, fy, cx, cy])]
                    self.image_list += [[images[i], images[i+1]]]
                    self.flow_list += [flows_forw[i]]
                    self.delta_list += [delta_forw[i]]
                    self.pose_list += [[poses[i], poses[i+1]]]
                    self.depth_list += [[depths[i], depths[i+1]]]

                    self.intrinsics_list += [np.array([fx, fy, cx, cy])]
                    self.image_list += [[images[i], images[i-1]]]
                    self.flow_list += [flows_back[i]]
                    self.delta_list += [delta_back[i]]
                    self.pose_list += [[poses[i], poses[i-1]]]
                    self.depth_list += [[depths[i], depths[i-1]]]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        intrinsics = self.intrinsics_list[index]

        image1 = cv2.imread(self.image_list[index][0])
        image2 = cv2.imread(self.image_list[index][1])

        poses = torch.as_tensor(self.pose_list[index])
        poses = poses.float()

        flow2d = frame_utils.read_gen(self.flow_list[index])[...,:2]
        delta = frame_utils.read_gen(self.delta_list[index])

        disp1 = frame_utils.read_gen(self.depth_list[index][0])
        disp2 = frame_utils.read_gen(self.depth_list[index][1])
        disp3 = disp1 + delta
        
        depth1 = torch.from_numpy(intrinsics[0] / disp1).float()
        depth2 = torch.from_numpy(intrinsics[0] / disp2).float()
        depth3 = torch.from_numpy(intrinsics[0] / disp3).float()

        s = 0.1 + .4 * np.random.rand()

        poses[:,:3] *= s
        depth1 = depth1 * s
        depth2 = depth2 * s
        depth3 = depth3 * s

        flowz = (1.0/depth3 - 1.0/depth1).unsqueeze(-1)
        flowxy = torch.from_numpy(flow2d).float()
        flowxyz = torch.cat([flowxy, flowz], dim=-1)

        intrinsics = torch.from_numpy(intrinsics).float()

        image1 = torch.from_numpy(image1).permute(2,0,1).float()
        image2 = torch.from_numpy(image2).permute(2,0,1).float()

        if self.augmentor is not None:
            image1, image2, depth1, depth2, flowxyz, intrinsics = \
                self.augmentor(image1, image2, depth1, depth2, flowxyz, intrinsics)

        # # exclude pixels at or near infinity
        # valid = (depth1 < 250.0).float()

        return image1, image2, poses, depth1, depth2, flowxyz, intrinsics


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