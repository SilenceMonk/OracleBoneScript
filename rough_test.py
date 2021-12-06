import timm
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm

from config import C
from utils.ema import ModelEMA
from utils.util import *

torch.backends.cudnn.benchmark = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

prRed(f"-> start test, rough stage, rank: {C.local_rank}")

# model
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

if C.ema:
    model = ema_model

# dataset
test_TF = transforms.Compose([
    transforms.Resize(size=[C.image_size, C.image_size], interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1626,), std=(0.3356,))
])

testset = datasets.ImageFolder(C.data, transform=test_TF)
testloader = DataLoader(dataset=testset,
                        batch_size=C.bs,
                        pin_memory=True)

# test
test_acc = AverageMeter()
tbar = tqdm(testloader, disable=C.local_rank != 0)
for x, y in tbar:
    x, y = x.to(device=C.device, non_blocking=True), y.to(device=C.device, non_blocking=True) // 4
    with torch.no_grad():
        lgt = model(x)
        ls = F.cross_entropy(lgt, y.long())
    _test_acc = accuracy(lgt, y)[0]
    test_acc.update(_test_acc)
    tbar.set_description('test_acc: {:.4f}, '.format(test_acc.avg))

prCyan(f'rough stage | test acc: {test_acc.avg}, rank: {C.local_rank}')
