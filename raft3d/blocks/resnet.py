
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models.utils import load_state_dict_from_url

MODEL_URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

import torch.utils.checkpoint as checkpoint


class FPN(ResNet):
    def __init__(self, output_dim=512, depth_input=False):
        super(FPN, self).__init__(Bottleneck, [3, 4, 6, 3], norm_layer=nn.BatchNorm2d)
        state_dict = load_state_dict_from_url(MODEL_URL)
        self.load_state_dict(state_dict)

        self.uconv1 = nn.Conv2d(2048, 512, 3, padding=1)
        self.uconv2 = nn.Conv2d(1024, 512, 3, padding=1)
        self.uconv3 = nn.Conv2d(512, output_dim, 1)

        self.norm1 = nn.BatchNorm2d(512)
        self.norm2 = nn.BatchNorm2d(512)
        
    def _forward_impl(self, x):
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        y = self.layer3(x)
        z = self.layer4(y)

        z = self.relu(self.uconv1(z))
        z = F.interpolate(z, x.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, z], dim=1)
        x = self.relu(self.uconv2(x))
        x = self.relu(self.uconv3(x))

        return x

    def forward(self, x):
        """ Input img, Output 1/8 feature map """
        return self._forward_impl(x)
