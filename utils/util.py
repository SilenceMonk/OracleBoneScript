import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn


# print functions
def prRed(prt): print("\033[91m{}\033[0m".format(prt))


def prGreen(prt): print("\033[92m{}\033[0m".format(prt))


def prYellow(prt): print("\033[93m{}\033[0m".format(prt))


def prLightPurple(prt): print("\033[94m{}\033[0m".format(prt))


def prPurple(prt): print("\033[95m{}\033[0m".format(prt))


def prCyan(prt): print("\033[96m{}\033[0m".format(prt))


def prRedWhite(prt): print("\033[41m{}\033[0m".format(prt))


def prWhiteBlack(prt): print("\033[7m{}\033[0m".format(prt))


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def init_weight(module, init_func=nn.init.kaiming_normal_, **kwargs):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            init_func(m.weight, **kwargs)
        elif isinstance(m, nn.Linear):
            init_func(m.weight, **kwargs)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def mixup_data(x, y, alpha=1.0, device='cuda:0'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device, non_blocking=True)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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


class CSVLogger:
    def __init__(self, args, fieldnames, filename='log.csv'):
        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()
