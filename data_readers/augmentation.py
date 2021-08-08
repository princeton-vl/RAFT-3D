import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

import math

class RGBDAugmentor:
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.5/3.14),
            transforms.ToTensor()])

    def spatial_transform(self, image1, image2, depth1, depth2, flow, intrinsics):
        
        max_scale = 0.6
        ht = image1.shape[1]
        wd = image1.shape[2]
        
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd)))
        
        scale = 2 ** np.random.uniform(min_scale, max_scale)
        ht1 = int(math.ceil(ht * scale))
        wd1 = int(math.ceil(wd * scale))

        fx, fy, cx, cy = intrinsics.unbind(dim=-1)
        sx = float(wd1) / float(wd)
        sy = float(ht1) / float(ht)
        intrinsics *= torch.as_tensor([sx, sy, sx, sy])

        image1 = F.interpolate(image1[None], [ht1, wd1], mode='bilinear', align_corners=True)[0]
        image2 = F.interpolate(image2[None], [ht1, wd1], mode='bilinear', align_corners=True)[0]
        depth1 = F.interpolate(depth1[None,None], [ht1, wd1], mode='bilinear', align_corners=True)[0,0]
        depth2 = F.interpolate(depth2[None,None], [ht1, wd1], mode='bilinear', align_corners=True)[0,0]

        flow = flow.permute(2,0,1)[None]
        flow = F.interpolate(flow, [ht1, wd1], mode='bilinear', align_corners=True)[0]
        flow = flow.permute(1,2,0) * torch.as_tensor([sx, sy, 1.0])

        y0 = np.random.randint(0, ht1 - self.crop_size[0] + 1)
        x0 = np.random.randint(0, wd1 - self.crop_size[1] + 1)

        image1 = image1[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        image2 = image2[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depth1 = depth1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depth2 = depth2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        intrinsics -= torch.as_tensor([0.0, 0.0, x0, y0])

        return image1, image2, depth1, depth2, flow, intrinsics
        
    def color_transform(self, image1, image2):
        """ Peform same perturbation over all images """
        wd = image1.shape[-1]
        image_stack = torch.cat([image1, image2], -1)
        image_stack = 255 * self.augcolor(image_stack / 255.0)
        return image_stack.split([wd, wd], -1)

    def __call__(self, image1, image2, depth1, depth2, flows, intrinsics):
        image1, image2 = self.color_transform(image1, image2)
        return self.spatial_transform(image1, image2, depth1, depth2, flows, intrinsics)


def resize_sparse_image(data, valid, ht1, wd1):
    ht, wd, dim = data.shape
    data = data.numpy()
    valid = valid.numpy() > 0.5

    coords = np.meshgrid(np.arange(wd), np.arange(ht))
    coords = np.stack(coords, axis=-1)

    coords0 = coords[valid]
    coords1 = coords0 * [ht1/float(ht), wd1/float(wd)]
    data1 = data[valid]

    xx = np.round(coords1[:,0]).astype(np.int32)
    yy = np.round(coords1[:,1]).astype(np.int32)

    v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
    xx = xx[v]
    yy = yy[v]

    data_resized = np.zeros([ht1, wd1, 3], dtype=np.float32)
    valid_resize = np.zeros([ht1, wd1], dtype=np.float32)

    data_resized[yy, xx] = data1[v]
    valid_resize[yy, xx] = 1.0

    data = torch.from_numpy(data_resized)
    valid = torch.from_numpy(valid_resize)

    return data, valid


class SparseAugmentor:
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.4/3.14),
            transforms.ToTensor()])


    def spatial_transform(self, image1, image2, depth1, depth2, flow, valid, intrinsics):
        
        max_scale = 0.5
        ht = image1.shape[1]
        wd = image1.shape[2]
        
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd)))

        
        scale = 2 ** np.random.uniform(min_scale, max_scale)
        
        ht1 = int(math.ceil(ht * scale))
        wd1 = int(math.ceil(wd * scale))

        if np.random.rand() < 0.8:
            fx, fy, cx, cy = intrinsics.unbind(dim=-1)
            sx = float(wd1) / float(wd)
            sy = float(ht1) / float(ht)
            
            intrinsics *= torch.as_tensor([sx, sy, sx, sy])

            image1 = F.interpolate(image1[None], [ht1, wd1], mode='bilinear', align_corners=False)[0]
            image2 = F.interpolate(image2[None], [ht1, wd1], mode='bilinear', align_corners=False)[0]
            depth1 = F.interpolate(depth1[None,None], [ht1, wd1], mode='nearest')[0,0]
            depth2 = F.interpolate(depth2[None,None], [ht1, wd1], mode='nearest')[0,0]

            flow, valid = resize_sparse_image(flow, valid, ht1, wd1)
            flow = flow * torch.as_tensor([sx, sy, 1])

        else:
            ht1 = ht
            wd1 = wd

        y0 = np.random.randint(0, ht1 - self.crop_size[0] + 20)
        x0 = np.random.randint(-50, wd1 - self.crop_size[1] + 50)

        y0 = np.clip(y0, 0, ht1 - self.crop_size[0])
        x0 = np.clip(x0, 0, wd1 - self.crop_size[1])

        image1 = image1[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        image2 = image2[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depth1 = depth1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depth2 = depth2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        intrinsics -= torch.as_tensor([0.0, 0.0, x0, y0])

        return image1, image2, depth1, depth2, flow, valid, intrinsics

    def color_transform(self, image1, image2):
        """ Peform same perturbation over all images """
        wd = image1.shape[-1]
        image_stack = torch.cat([image1, image2], -1)
        image_stack = 255 * self.augcolor(image_stack / 255.0)
        return image_stack.split([wd, wd], -1)

    def __call__(self, image1, image2, depth1, depth2, flow, valid, intrinsics):
        image1, image2 = self.color_transform(image1, image2)
        return self.spatial_transform(image1, image2, depth1, depth2, flow, valid, intrinsics)
