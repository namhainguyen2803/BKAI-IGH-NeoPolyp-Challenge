from tqdm import tqdm
from utils import *
from models import *
from dataloader import *
from criterion import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict
from torch.utils.data import DataLoader, random_split
import wandb
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:
        input = input.to(DEVICE)
        target = target.to(DEVICE)

        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            dice = dice_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            dice = dice_score(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['dice'].update(dice.item(), input.size(0))
        postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg)
            ])

def validate(config, val_loader, model, criterion):
    avg_meters = {"loss": AverageMeter(),
                  "dice": AverageMeter()}
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
                dice = dice_score(output, target)
            else:
                output = model(input)
                loss = criterion(output, target)
                dice = dice_score(output, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['dice'].update(dice.item(), input.size(0))
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg)
            ])

def initialize(config):

    if config['model'] == 'UNet':
        model = UNet(config['num_classes'],config['input_channels'])
    elif config['model'] == 'NestedUNet':
        model = NestedUNet(config['num_classes'],config['input_channels'],config['deep_supervision'])
    if config['model'] == 'PretrainedUNet':
        model = PretrainedUNet(num_classes=config['num_classes'],in_channels=config['input_channels'])
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

    if config['loss'] == 'CEDiceLoss':
        weights = torch.Tensor([[0.4, 0.55, 0.05]]).to(DEVICE)
        criterion = CEDiceLoss(weights).to(DEVICE)
    elif config['loss'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().to(DEVICE)
    else:
        raise NotImplementedError

    return model.to(DEVICE), optimizer, scheduler, criterion.to(DEVICE)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hello')

    # Add arguments
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--deep_supervision', action='store_true', help='Use deep supervision for UNet++')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer choice')
    parser.add_argument('--model', type=str, default='PretrainedUNet', help='Model architecture')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--loss', type=str, default='CEDiceLoss', help='Loss function')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', help='Learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/model.pth', help='Path to save checkpoints')

    args = parser.parse_args()

    return args

def main():

    IMAGE_TRAINING_PATH = "dataset/train/train"
    IMAGE_GT_PATH = "dataset/train_gt/train_gt"
    IMAGE_TESTING_PATH=  "dataset/test/test"

    CHECKPOINT_DIR = "checkpoint"

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    TRAIN_SIZE = 0.8
    VALID_SIZE = 0.2

    config = vars(parse_arguments())


    train_loss_array = list()
    test_loss_array = list()
    best_loss = -1e9
    use_wandb = True
    trigger = 0

    if use_wandb:
        wandb.login(
            key="42a6113f1cb8bfc726e81b54b5b43967cdb2a437",
        )
        wandb.init(
            project="PolypSegment"
        )

    unet_dataset = UNetDataClass(IMAGE_TRAINING_PATH, IMAGE_GT_PATH)

    train_set, valid_set = random_split(unet_dataset,
                                        [int(TRAIN_SIZE * len(unet_dataset)),
                                         int(VALID_SIZE * len(unet_dataset))])

    train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=config["batch_size"], shuffle=True)

    model, optimizer, scheduler, criterion = initialize(config)

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        train_log = train(config, train_dataloader, model, criterion, optimizer)
        val_log = validate(config, valid_dataloader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        trigger += 1

        if val_log["loss"] < best_loss:
            checkpoint = {
                "model_name": config["model"],
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "num_classes": config["num_classes"],
                "deep_supervision": config["deep_supervision"],
                "input_channels": config["input_channels"]
            }

            torch.save(checkpoint, config["checkpoint_path"])

            best_loss = val_log["loss"]
            print("=> saved best model")
            trigger = 0

        # early stopping
        if 0 <= config['early_stopping'] <= trigger:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

        train_loss_array.append(train_log["loss"])
        test_loss_array.append(val_log["loss"])

        if use_wandb:
            wandb.log({"Train loss": train_log["loss"], "Valid loss": val_log["loss"]})

        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()