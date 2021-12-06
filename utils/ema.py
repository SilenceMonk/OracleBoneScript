import pdb
from copy import deepcopy

import torch

"""class ModelEMA(object):
    def __init__(self, args, model, alpha=0.9):
        self.ema = model
        self.ema.to(args.device)
        self.ema.eval()
        self.alpha = alpha
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def __call__(self, x):
        return self.ema(x)

    def update(self, model):
        for ema_param, param in zip(self.ema.parameters(), model.parameters()):
            ema_param.data.mul_(self.alpha).add_(1 - self.alpha, param.data)"""


class ModelEMA(object):
    def __init__(self, args, model, alpha=0.99):
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.alpha = alpha
        self.ema_has_module = hasattr(self.ema, 'module')
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def __call__(self, x):
        return self.ema(x)

    def eval(self):
        self.ema.eval()

    def state_dict(self):
        return self.ema.state_dict()

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                delta = model_v - ema_v
                ema_v = model_v - self.alpha * delta
                esd[k].copy_(ema_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])



