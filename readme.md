# Oracle Bone Script Recognition 

## Requirements
```text
torch
tensorboard
timm
pyyaml
tqdm
```

## File Tree
```text
│  base.py
│  config.py
│  dataset.py
│  fine.py
│  readme.md
│  rough.py
│          
├─cfgs
│      fine_resnet18_pretrained.yaml
│      fine_resnet18_raw.yaml
│      rough_resnet18_pretrained.yaml
│      rough_resnet18_raw.yaml
│      
├─ckpt
├─data
│  └─obs
│      ├─0102
│      │      person_0000.jpg
│      │      person_0001.jpg
│      │      ...
│      │      
│      ├─0103
│      │      person_0000.jpg
│      │      person_0001.jpg
│      │      ...
│      ...
│              
├─logs
│
│          
├─pretrained_ecoder_weights
│  └─checkpoints
│          resnet18-5c106cde.pth
│          
├─scripts
│      local.sh
│      normal.sh
│      slurm.sh
│      
├─utils
│  │  ddp.py
│  │  ema.py
│  │  util.py
│  |  init__.py
```

## Benchmark
* arch
  * ResNet18
* method: 
  * loss: standard cross entropy 
  * optimizer: RAdam (https://paperswithcode.com/method/radam)
  * lr_scheduler: cosine annealing 
  * data augmentation: 
    * affine
    * normalization
    * mixup* (in fine-grained stage*, https://arxiv.org/pdf/1710.09412.pdf)
  * exponential moving average
* __rough__ classes: 10, __fine__-grained classes: 40
* valset: random 20% hold out, repeated 3 times

| stage | ImageNet pretrained |val acc (mean/std)|
| --- | --- | --- |
| rough| x |99.9349/0.0921|
| rough| |__100/0.0__|
| fine| x |96.6145/0.6037|
| fine| |__97.4524/0.4260__|


## Training

### 1. modify config file in cfgs
* example: _cfgs/fine_resnet18_pretrained.yaml_
```yaml
# Experiment Config
name : fine_resnet18_pretrained.yaml # logger's name
data : data/obs/ # the dir where data is stored
logs : logs/ # the dir training logs will be saved under
arch : resnet18 # support timm models
pretrained : Ture # (bool) use ImageNet pretrained weights or not
ema : Ture # (bool) exponential moving average

resume : # 'logs/2021_11_18_20_04_20/best.pth' # resume training from giving dir
torch_hub_dir : pretrained_ecoder_weights # the dir where ImageNet pretrained weights are stored
local_rank : 0
world_size : 1
val_ratio: 0.2 # validation set ratio

# Image Config
num_classes : 40
image_size : 128

# Train Config
base_lr : 0.000015625 # lr = base_lr * bs * world_size (linear scaling by actual bs)
optimizer : RAdam # support: optimizers in timm and torch, (RAdam: https://paperswithcode.com/method/radam)
weight_decay : 0.001
nesterov : False # (bool) only for SGD
momentum : 0.9 # only for SGD
nepochs : 50 # total training epochs
```

### 2. modify bash script
* example: _scripts/local.sh_
```shell
export cfg_dir=cfgs/fine_resnet18_pretrained.yaml # config file dir
export ngpu=1 bs=64 ddp=~ # set number of gpus, batchsize per gpu, distributed mode (normal, slurm or ~ (no ddp))
export seed=42 # reset random seed for reproduction
python fine.py
```
### 3. run script
```shell
sh scripts/local.sh
```

### * reproduce benchmark result
use default config files and scripts

### * customize training loop
modify _base.py_ or create a new one for new algorithms

## Testing (benchmark)
### 1. download pretrained weights above
https://pan.baidu.com/s/12okB6FhpmVXOMwsgXrw20g \
key: qfg3
### 2. create model, load weights using timm
```python
import torch
import timm

# rough
rough_model = timm.create_model(model_name='resnet18', num_classes=10)
ckpt = torch.load('ckpt/rough_resnet18_pretrained.pth')
rough_model.load_state_dict(ckpt['ema'])

# fine
fine_model = timm.create_model(model_name='resnet18', num_classes=40)
ckpt = torch.load('ckpt/fine_resnet18_pretrained.pth')
fine_model.load_state_dict(ckpt['ema'])
```

### 3. preprocess input as in validation
```python
from torchvision.transforms import transforms

test_TF = transforms.Compose([
    transforms.Resize(size=128),
    transforms.Normalize(mean=(0.1626,), std=(0.3356,))
])
```
### 4. test loop ...