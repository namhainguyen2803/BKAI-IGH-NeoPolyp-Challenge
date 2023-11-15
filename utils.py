import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _to_one_hot(y, num_classes):
    """
    Convert a categorical tensor to one-hot encoded tensor
    :param y: must be in type int or long, can be tensor, matrix, vector
    :param num_classes: number of categories
    :return: one-hot encoded vector for each element of y
    """
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype).to(DEVICE)

    return zeros.scatter(scatter_dim, y_tensor, 1)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weights_init(model):
    if isinstance(model, nn.Linear):
        # Xavier Distribution
        torch.nn.init.xavier_uniform_(model.weight)

def save_model(model, optimizer, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def dice_score(output, y_target):
    """
    Handle dice score for multi-classes case:

        if red segment then y_target == 1
        if green segment then y_target == 2
        if background then y_target == 0

    :param output: (N, C, H, W)
    :param y_target: (N, H, W)
    :return:
    """

    y_predict = torch.argmax(output, dim=1)
    one_hot_y_pred = _to_one_hot(y_predict, 3)
    one_hot_y_target = _to_one_hot(y_target, 3)

    intersection = torch.sum(one_hot_y_pred * one_hot_y_target, dim=1)
    cardinality = torch.sum(one_hot_y_pred + one_hot_y_target, dim=1)

    dice_score = (2 * intersection + 1e-6) / (cardinality + 1e-6)
    return torch.mean(dice_score)

