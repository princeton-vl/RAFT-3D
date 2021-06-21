import torch
import cv2
from torch.utils.tensorboard import SummaryWriter


SUM_FREQ = 100

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def normalize_image(image):
    image = image[:, [2,1,0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=image.device)
    return (image/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Logger:
    def __init__(self):
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        if self.writer is None:
            self.writer = SummaryWriter()

        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}] ".format(self.total_steps+1)
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        for key in self.running_loss:
            val = self.running_loss[key] / SUM_FREQ
            self.writer.add_scalar(key, val, self.total_steps)
            self.running_loss[key] = 0.0

    def push(self, metrics):

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1