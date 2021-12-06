import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torchvision.transforms import transforms, InterpolationMode
import torchvision.datasets as datasets
from config import C


general_TF = transforms.Compose([
    transforms.Resize(size=[C.image_size, C.image_size], interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

train_TF = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.Normalize(mean=C.image_mean, std=C.image_mean)
])

val_TF = transforms.Compose([
    transforms.Normalize(mean=C.image_mean, std=C.image_mean)
])


print(f"-> prepare dataset, rank: {C.local_rank}")

dataset = datasets.ImageFolder(C.data, transform=general_TF)
C.num_val = int(len(dataset) * C.val_ratio)
C.num_train = len(dataset) - C.num_val
trainset, valset = torch.utils.data.random_split(dataset, [C.num_train, C.num_val])

sampler = DistributedSampler if C.ddp in ['slurm', 'normal'] else RandomSampler

trainloader = DataLoader(dataset=trainset,
                         batch_size=C.bs,
                         sampler=sampler(trainset),
                         # drop_last=True,
                         pin_memory=True)

valloader = DataLoader(dataset=valset,
                       batch_size=C.bs,
                       sampler=sampler(valset),
                       # drop_last=True,
                       pin_memory=True)

C.niters_per_epoch = len(trainloader)
print(f"-> done, rank: {C.local_rank}, total samples: "
      f"{C.niters_per_epoch * C.bs * C.world_size} | {len(valloader) * C.bs * C.world_size}")
