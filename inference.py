import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, PILToTensor, ToPILImage, Compose, InterpolationMode
import os
from dataloader import UNetTestDataClass
from models import *

PREDICTED_MASK_PATH = "predicted_masks"
IMAGE_TESTING_PATH = "dataset/test/test"
CHECKPOINT_PATH = 'checkpoint/model.pth'

loaded_checkpoint = torch.load(CHECKPOINT_PATH)
model_name = loaded_checkpoint['model_name']
last_epoch = loaded_checkpoint['epoch']
num_classes = loaded_checkpoint['num_classes']
inp_channels = loaded_checkpoint['input_channels']
deep_supervision = loaded_checkpoint['deep_supervision']

if model_name == 'UNet':
    model = UNet(num_classes, inp_channels)
elif model_name == 'NestedUNet':
    model = NestedUNet(num_classes, inp_channels, deep_supervision)
if model_name == 'PretrainedUNet':
    model = PretrainedUNet(num_classes=num_classes, in_channels=inp_channels)
else:
    raise NotImplementedError

model.load_state_dict(loaded_checkpoint['model_state_dict'])
model.eval()

unet_test_dataset = UNetTestDataClass(IMAGE_TESTING_PATH)
test_dataloader = DataLoader(unet_test_dataset, batch_size=2, shuffle=True)

if not os.path.isdir(PREDICTED_MASK_PATH):
    os.mkdir(PREDICTED_MASK_PATH)
for _, (img, path, H, W) in enumerate(test_dataloader):
    a = path
    b = img
    h = H
    w = W

    with torch.no_grad():
        predicted_mask = model(b)
    for i in range(len(a)):
        image_id = a[i].split('/')[-1].split('.')[0]
        filename = image_id + ".png"
        mask2img = Resize((h[i].item(), w[i].item()), interpolation=InterpolationMode.NEAREST)(
            ToPILImage()(F.one_hot(torch.argmax(predicted_mask[i], 0)).permute(2, 0, 1).float()))
        mask2img.save(os.path.join(PREDICTED_MASK_PATH, filename))


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)


def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:, :, ::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:, :, channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r


MASK_DIR_PATH = PREDICTED_MASK_PATH  # change this to the path to your output mask folder
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(r'output.csv', index=False)