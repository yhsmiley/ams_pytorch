import torch
import numpy as np
from torchvision import transforms

from .yolov4_dataset import YOLOv4_Dataset


def fetch_dataloader(params, teacher_model):
    """
    Fetch and return train/dev dataloader with hyperparameters
    """
    trainset = YOLOv4_Dataset(root_dir='./data-imagenet/train', model=teacher_model, params=params, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size//params.subdivisions, shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda, drop_last=True)

    devset = YOLOv4_Dataset(root_dir='./data-imagenet/test', model=teacher_model, params=params, train=False)
    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size//params.subdivisions, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda, collate_fn=val_collate)

    return trainloader, devloader

def val_collate(batch):
    return tuple(zip(*batch))
