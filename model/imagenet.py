import cv2
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageNet(Dataset):
    """ImageNet dataset from extracted ImageNet Images"""

    def __init__(self, root_dir, model=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            model (callable): Model to be applied on a sample to get label.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.model = model
        self.transform = transform

        self.model.eval()

        # put all images into np array
        files = glob.glob(self.root_dir + "/*")
        filesize = (224,224)        # imagenet input size
        self.data = np.array([cv2.resize(cv2.imread(file), filesize) for file in files])
        data_transformed = self.data.copy()
        self.data = self.data.transpose((0, 3, 1, 2))
        # self.data = torch.tensor(self.data)
        print(f'data shape: {self.data.shape}')

        # perform test transforms
        dev_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        data_transformed = torch.stack([dev_transformer(Image.fromarray(img)) for img in data_transformed])

        # get all labels
        self.targets = []
        bs = 16
        data_split = torch.split(data_transformed, bs)
        for data_smol in data_split:
            teacher_input = data_smol.float().cuda()
            teacher_output = self.model(teacher_input)
            # print(f'teacher shape: {teacher_output.shape}')
            teacher_output = torch.argmax(teacher_output, dim=1)
            # print(f'teacher output: {teacher_output}')
            self.targets.extend(np.array(teacher_output.cpu()))
        self.targets = torch.tensor(self.targets)
        print(f'targets shape: {self.targets.shape}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[idx], self.targets[idx]

        img = img.transpose((1, 2, 0))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
