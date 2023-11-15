import random

import numpy as np
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SplitDataset:

    def __init__(self, training_images_path=None, training_masks_path=None, testing_images_path=None):
        super(SplitDataset).__init__()

        self.images_test_list = None
        self.masks_list = None
        self.images_list = None
        self.images_path = training_images_path
        self.masks_path = training_masks_path
        self.test_images_path = testing_images_path

    def create_training_set(self):
        assert self.masks_path is not None
        assert self.images_path is not None
        images_list = os.listdir(self.images_path)
        masks_list = os.listdir(self.masks_path)

        self.images_list = [os.path.join(self.images_path, image_name) for image_name in images_list]
        self.masks_list = [os.path.join(self.masks_path, mask_name) for mask_name in masks_list]

        image_mask_pairs = list(zip(self.images_list, self.masks_list))
        random.shuffle(image_mask_pairs)
        split_ratio = 0.8
        split_index = int(len(image_mask_pairs) * split_ratio)
        train_pairs = image_mask_pairs[:split_index]
        test_pairs = image_mask_pairs[split_index:]

        train_images_list, train_masks_list = zip(*train_pairs)
        valid_images_list, valid_masks_list = zip(*test_pairs)

        training_set = UNetDataClass(purpose="train", images_list=train_images_list, masks_list=train_masks_list)
        validation_set = UNetDataClass(purpose="validation", images_list=valid_images_list, masks_list=valid_masks_list)
        return training_set, validation_set

    def create_testing_set(self):
        assert self.test_images_path is not None
        images_list = os.listdir(self.test_images_path)

        self.images_test_list = [os.path.join(self.test_images_path, image_name) for image_name in images_list]

        test_set = UNetDataClass(purpose="test", images_list=self.images_test_list)
        return test_set


class UNetDataClass(Dataset):
    def __init__(self, purpose, images_list, masks_list=None, resize=(256,256)):
        super(UNetDataClass, self).__init__()

        self.transform = None
        self.purpose = purpose
        self.images_list = images_list
        self.masks_list = masks_list
        self.resize = resize

        self.smart_transform()

    def smart_transform(self):

        if self.purpose == "train":
            self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomGamma(gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
                        A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2(),
                    ])
        else:
            self.transform = A.Compose([
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2(),
                    ])
    def __getitem__(self, idx):
        
        if self.purpose == "train" or self.purpose == 'validation':
            img_path = self.images_list[idx]
            label_path = self.masks_list[idx]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = self.read_mask(label_path)
            image = cv2.resize(image, self.resize)

            transformed = self.transform(image=image, mask=label)
            image = transformed["image"].float()
            label = transformed["mask"].float()
            label = label.permute(2, 0, 1)

            return image, label

        elif self.purpose == "test":
            img_path = self.images_list[idx]
            ori_img = cv2.imread(img_path)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            ori_w = ori_img.shape[0]
            ori_h = ori_img.shape[1]
            img = cv2.resize(ori_img, self.resize)
            input_img = self.transform(image=img)["image"]
            return input_img, img_path, ori_h, ori_w

    def read_mask(self, mask_path):
        # Read the image from the specified path using OpenCV
        image = cv2.imread(mask_path)

        # Resize the image to the specified dimensions
        image = cv2.resize(image, self.resize)

        # Convert the image to the HSV color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color range values for the lower and upper boundaries of red color
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 20])
        upper2 = np.array([179, 255, 255])

        # Create masks for the lower and upper boundaries of red color
        lower_mask = cv2.inRange(image, lower1, upper1)
        upper_mask = cv2.inRange(image, lower2, upper2)

        # Combine the masks to get a full red mask
        red_mask = lower_mask + upper_mask
        red_mask[red_mask != 0] = 1

        # Define color range values for the green color
        green_mask = cv2.inRange(image, (36, 25, 25), (70, 255, 255))

        # Set non-zero values in the green mask to 2
        green_mask[green_mask != 0] = 2

        # Combine the red and green masks using bitwise OR
        full_mask = cv2.bitwise_or(red_mask, green_mask)

        # Expand the dimensions of the mask to add a channel dimension
        full_mask = np.expand_dims(full_mask, axis=-1)

        # Convert the mask to uint8 type
        full_mask = full_mask.astype(np.uint8)

        # Return the final mask
        return full_mask

    def __len__(self):
        return len(self.images_list)