from tqdm import tqdm
from collections import OrderedDict
from utils import *
from models import *
from dataloader import *
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, random_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:
        input = input.to(DEVICE)
        target = target.to(DEVICE)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.to(DEVICE)
            target = target.to(DEVICE)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def initialize(config):

    if config['model'] == 'UNet':
        model = UNet(config['num_classes'],config['input_channels'])
    elif config['model'] == 'NestedUNet':
        model = NestedUNet(config['num_classes'],config['input_channels'],config['deep_supervision'])
    else:
        raise NotImplementedError

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    else:
        raise NotImplementedError

    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    return model.to(DEVICE), optimizer, scheduler, criterion.to(DEVICE)

def main():

    IMAGE_TRAINING_PATH = "dataset/train/train"
    IMAGE_GT_PATH = "dataset/train_gt/train_gt"

    TRAIN_SIZE = 0.8
    VALID_SIZE = 0.2
    BATCH_SIZE = 12

    config = {
        "epochs": 2,
        "deep_supervision": False,
        "num_classes": 3,
        "input_channels": 3,
        "optimizer": "SGD",
        "model": "UNet",
        "min_lr": 1e-5,
        "lr": 1e-3,
        "loss": "BCEWithLogitsLoss",
        "scheduler": "CosineAnnealingLR"
    }

    unet_dataset = UNetDataClass(IMAGE_TRAINING_PATH, IMAGE_GT_PATH)
    unet_dataset = UNetDataClass(IMAGE_TRAINING_PATH, IMAGE_GT_PATH)

    train_set, valid_set = random_split(unet_dataset,
                                        [int(TRAIN_SIZE * len(unet_dataset)) ,
                                         int(VALID_SIZE * len(unet_dataset))])

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)

    model, optimizer, scheduler, criterion = initialize(config)

    best_iou = 0
    trigger = 0
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_dataloader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, valid_dataloader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()