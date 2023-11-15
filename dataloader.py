import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Resize, Compose, InterpolationMode

class UNetDataClass(Dataset):
    def __init__(self, images_path, masks_path):
        super(UNetDataClass, self).__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = None

        self.smart_init()

    def smart_init(self):
        images_list = os.listdir(self.images_path)
        masks_list = os.listdir(self.masks_path)

        self.transform = Compose([Resize((800, 1120), interpolation=InterpolationMode.BILINEAR),
                                  transforms.ToTensor()])

        self.images_list = [os.path.join(self.images_path, image_name) for image_name in images_list]
        self.masks_list = [os.path.join(self.masks_path, mask_name) for mask_name in masks_list]

    def __getitem__(self, index):
        img_path = self.images_list[index]
        mask_path = self.masks_list[index]

        data = Image.open(img_path)
        label = Image.open(mask_path)

        data = self.transform(data)
        label = self.transform(label)

        label = torch.where(label > 0.65, 1.0, 0.0)  # binarize

        label[2, :, :] = 0.0001
        # if background then 2
        # if red segment then 0
        # if green segment then 1
        label = torch.argmax(label, 0).type(torch.int64)

        return data, label

    def __len__(self):
        return len(self.images_list)


class UNetTestDataClass(Dataset):
    def __init__(self, images_path):
        super(UNetTestDataClass, self).__init__()

        self.transform = None
        self.images_list = None
        self.images_path = images_path
        self.init_smart()

    def init_smart(self):
        images_list = os.listdir(self.images_path)
        self.images_list = [os.path.join(self.images_path, image_name) for image_name in images_list]
        self.transform = Compose([Resize((800, 1120), interpolation=InterpolationMode.BILINEAR),
                                  transforms.ToTensor()])

    def __getitem__(self, index):
        img_path = self.images_list[index]
        data = Image.open(img_path)
        h = data.size[1]
        w = data.size[0]
        data = self.transform(data)
        return data, img_path, h, w

    def __len__(self):
        return len(self.images_list)