# encoding: utf-8
import os

import torch
import yaml
from easydict import EasyDict as edict

from utils.ddp import setup_slurm_distributed, setup_distributed
from utils.util import reset_seed, prGreen, prCyan

# parse configs
cfg_dir = os.environ['cfg_dir']
f = open(cfg_dir, 'r', encoding="utf-8")
C = edict(yaml.load(f, Loader=yaml.FullLoader))

C.ngpu = int(os.environ['ngpu'])
C.bs = int(os.environ['bs'])
C.device = torch.device('cuda', C['local_rank'])
C.ddp = os.environ['ddp']
C.seed = int(os.environ['seed'])

torch.hub.set_dir(C['torch_hub_dir'])
reset_seed(C['seed'])

if C.ddp == 'slurm':
    prCyan("-> distributed! mode: slurm")
    setup_slurm_distributed(C)
elif C.ddp == 'normal':
    prCyan("-> distributed! mode: normal")
    setup_distributed(C)
else:
    prCyan("-> not distributed!")
    C.ddp = None

C.lr = C.base_lr * C.bs * C.world_size

# stat from https://github.com/Olafyii/Oracle-Bone-Characters/blob/master/utils.py
C.image_mean = (0.1626,)
C.image_std = (0.3356,)
prGreen(f"-> config: \n{C}")

