import time

import timm
import torch.distributed as dist
import torch.nn.functional as F
import timm.optim as timm_opt
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import C
from utils.dataset import trainloader, valloader, train_TF, val_TF
from utils.ema import ModelEMA
from utils.util import *

torch.backends.cudnn.benchmark = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print(f"-> start fine stage, rank: {C.local_rank}")
if C.local_rank == 0:
    # logger
    C.log_root = C.logs + time.strftime('%Y_%m_%d+%H_%M_%S', time.localtime()) + '/'
    mkdir(C.log_root)
    C.csv_logger = CSVLogger(args=C,
                             fieldnames=['epoch', 'train_ls', 'train_acc',
                                         'val_ls', 'val_fine_acc', 'val_rough_acc',
                                         'best_acc'],
                             filename=C.log_root + 'log.csv')

    C.summary_writer = SummaryWriter(C.log_root)

# model, optimizer, scheduler
model = timm.create_model(C.arch, pretrained=C.pretrained, num_classes=C.num_classes)

if not C.pretrained:
    init_weight(model)

if C.ddp in ['slurm', 'normal']:
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device=C.device)
    model = DDP(model, device_ids=[C.local_rank], output_device=C.local_rank)
else:
    model.to(device=C.device)

if C.ema:
    ema_model = ModelEMA(C, model)
    val_model = ema_model
else:
    val_model = model

if C.optimizer == 'SGD':
    optimizer = optim.SGD(params=model.parameters(),
                          lr=C.lr,
                          momentum=C.momentum,
                          weight_decay=C.weight_decay,
                          nesterov=C.nesterov)
elif C.optimizer == 'RAdam':
    optimizer = getattr(timm_opt, C.optimizer)(params=model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
else:
    optimizer = getattr(optim, C.optimizer)(params=model.parameters(), lr=C.lr)

scheduler = CosineAnnealingLR(optimizer, T_max=C.nepochs)

gs = 0  # global step
plateau_count = 0

# resume from ckpt
if C.resume:
    assert os.path.isfile(C.resume), f'pretrained model pth does not exist: {C.resume}, ' \
                                     f'rank: {C.local_rank}'
    print(f"-> loading pretrained model '{C.resume}', rank: {C.local_rank}")
    ckpt_dict = torch.load(C.resume)
    if C.ddp in ['slurm', 'normal']:
        model.module.load_state_dict(ckpt_dict['model'])
    else:
        model.load_state_dict(ckpt_dict['model'])
    if C.ema:
        ema_model.ema.load_state_dict(ckpt_dict['ema'])
    scheduler.load_state_dict(ckpt_dict['scheduler'])
    optimizer.load_state_dict(ckpt_dict['optimizer'])

train_ls = AverageMeter()
train_acc = AverageMeter()
val_ls = AverageMeter()
val_fine_acc = AverageMeter()
val_rough_acc = AverageMeter()

best_fine_acc = 0
best_rough_acc = 0
best_epoch = 0
for epoch in range(C.nepochs):
    # train
    train_ls.reset()
    train_acc.reset()
    model.train()
    tbar = tqdm(trainloader, disable=C.local_rank != 0)
    if C.ddp in ['slurm', 'normal']:
        trainloader.sampler.set_epoch(epoch)
    for x, y in tbar:
        x, y = x.to(device=C.device, non_blocking=True), y.to(device=C.device, non_blocking=True)
        x = train_TF(x)
        """lgt = model(x)
        ls = F.cross_entropy(lgt, y.long())"""
        mixed_x, y_a, y_b, lam = mixup_data(x, y, device=C.device)
        lgt = model(mixed_x)
        ls = mixup_criterion(F.cross_entropy, lgt, y_a, y_b, lam)
        ls.backward()
        gs += 1
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if C.ema:
            ema_model.update(model)
        # scheduler.step()
        if C.local_rank == 0:
            _train_acc = accuracy(lgt, y)[0]
            train_ls.update(ls.item())
            train_acc.update(_train_acc)
            tbar.set_description('epoch: {}, '
                                 'ls_s: {:.4f}, '
                                 'train_acc: {:.4f}, '
                                 'lr: {:.4e}'.format(epoch,
                                                     train_ls.avg,
                                                     train_acc.avg,
                                                     # optimizer.state_dict()['param_groups'][0]['lr']
                                                     scheduler.get_last_lr()[0]
                                                     ))

            C.summary_writer.add_scalar('train_ls', train_ls.avg, gs)
            C.summary_writer.add_scalar('train_acc', train_acc.avg, gs)

    # val
    val_ls.reset()
    val_fine_acc.reset()
    val_rough_acc.reset()
    val_model.eval()
    vbar = tqdm(valloader, disable=C.local_rank != 0)
    for x, y in vbar:
        x, y = x.to(device=C.device, non_blocking=True), y.to(device=C.device, non_blocking=True)
        x = val_TF(x)
        with torch.no_grad():
            lgt = val_model(x)
            ls = F.cross_entropy(lgt, y.long())
        _val_fine_acc = accuracy(lgt, y)[0]
        rough_pred_y, rough_y = lgt.argmax(dim=1) // 4, y // 4
        _val_rough_acc = (rough_pred_y == rough_y).sum() / C.bs * 100
        # _val_rough_acc = accuracy(lgt.argmax(dim=1) // 4,  y // 4)[0]
        val_ls.update(ls.item())
        val_fine_acc.update(_val_fine_acc)
        val_rough_acc.update(_val_rough_acc)
        vbar.set_description('epoch: {}, '
                             'val_ls: {:.4f}, '
                             'val_acc: {:.4f}/{:.4f} '.format(epoch,
                                                              val_ls.avg,
                                                              val_fine_acc.avg,
                                                              val_rough_acc.avg))

    if C.ddp in ['slurm', 'normal']:
        # _val_acc = torch.tensor(val_acc.avg, device=C.device)
        dist.all_reduce(val_fine_acc.avg / C.world_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_rough_acc.avg / C.world_size, op=dist.ReduceOp.SUM)
        # val_acc.avg = _val_acc.item()

    if C.local_rank == 0:
        model_sd = model.module.state_dict() if C.ddp in ['slurm', 'normal'] else model.state_dict()
        ema_sd = ema_model.state_dict() if C.ema else None

        ckpt_dict = {'config': C,
                     'epoch': epoch,
                     'model': model_sd,
                     'ema': ema_sd,
                     'scheduler': scheduler.state_dict(),
                     'optimizer': optimizer.state_dict()}

        torch.save(ckpt_dict, C.log_root + 'rough_resnet18_raw.pth')
        # save according to fine grained acc
        if val_fine_acc.avg > best_fine_acc:
            torch.save(ckpt_dict, C.log_root + 'fine_resnet18_pretrained.pth')
            prCyan(f"epoch: {epoch}, val_acc: {val_fine_acc.avg}/{val_rough_acc.avg}, best ckpt saved")
            best_fine_acc = val_fine_acc.avg
            best_rough_acc = val_rough_acc.avg
            best_epoch = epoch
            plateau_count = 0
        else:
            plateau_count += 1
            prGreen("epoch: {}, val_acc: {:.4f}/{:.4f} | p{:02d}".format(epoch,
                                                                         val_fine_acc.avg,
                                                                         val_rough_acc.avg,
                                                                         plateau_count))

        C.summary_writer.add_scalar('val_ls', val_ls.avg, gs)
        C.summary_writer.add_scalar('val_fine_acc', val_fine_acc.avg, gs)
        C.summary_writer.add_scalar('val_rough_acc', val_rough_acc.avg, gs)

        row = {'epoch': str(epoch),
               'train_ls': '{:.4f}'.format(train_ls.avg),
               'train_acc': '{:.4f}'.format(train_acc.avg),
               'val_ls': '{:.4f}'.format(val_ls.avg),
               'val_fine_acc': '{:.4f}'.format(val_fine_acc.avg),
               'val_rough_acc': '{:.4f}'.format(val_rough_acc.avg),
               'best_acc': '{:.4f}/{:.4f}'.format(best_fine_acc, best_rough_acc)}
        C.csv_logger.writerow(row)

    """if plateau_count > C.max_plateau:
        prWhiteBlack(f'plateau_count ({plateau_count}) > max_plateau ({C.max_plateau}), break')
        break"""
    scheduler.step()
    # scheduler.step(100 - val_acc.avg)

if C.local_rank == 0:
    C.csv_logger.writerow({'epoch': str(best_epoch),
                           'best_acc': '{:.4f}/{:.4f}'.format(best_fine_acc, best_rough_acc)})
    C.csv_logger.close()
    C.summary_writer.close()
    prPurple(f'best acc: {best_fine_acc}/{best_rough_acc}, epoch: {best_epoch}')
