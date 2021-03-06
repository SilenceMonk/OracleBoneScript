# Oracle Bone Script Recognition Benchmark

## Requirements
```text
torch
tensorboard
timm
pyyaml
tqdm
```

## File Tree
### _OracalBoneScript-main_
```text
│  base.py
│  fine.py
│  rough.py
│  config.py
│  readme.md
│          
├─cfgs
│      fine_resnet18_pretrained.yaml
│      fine_resnet18_raw.yaml
│      rough_resnet18_pretrained.yaml
│      rough_resnet18_raw.yaml
|             
├─logs
│
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
│  |  dataset.py
│  |  __init__.py
```

### _OBS-extend_
* https://pan.baidu.com/s/1naYjcgtUyxuVkMhGaAy7jg
* key: pyyg

```text
├─pretrained_ecoder_weights
│  └─checkpoints
│          resnet18-5c106cde.pth      
├─ckpt
│      fine_resnet18_pretrained.pth
│      rough_resnet18_pretrained.pth
├─data
│  └─obs
│      └─train
|          ├─0102
|          │      person_0000.jpg
|          │      person_0001.jpg
|          │      ...
|          │      
|          ├─0103
|          │      person_0000.jpg
|          │      person_0001.jpg
|          │      ...
|          ...
```
## Installation
* download [OBS-extend](#obs-extend) files 
* put them under [OracalBoneScript-main](#oracalbonescript-main)

## Result
* arch
  * ResNet18
* method: 
  * loss: standard cross entropy 
  * optimizer: RAdam (https://paperswithcode.com/method/radam)
  * lr_scheduler: cosine annealing 
  * data augmentation: 
    * affine
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
* example: [_cfgs/fine_resnet18_pretrained.yaml_](cfgs/fine_resnet18_pretrained.yaml)
```yaml
# Experiment Config
name : fine_resnet18_pretrained.yaml # logger's name
data : data/obs/train # the dir where trainset is stored
logs : logs/ # the dir training logs will be saved under
arch : resnet18 # support timm models
pretrained : Ture # (bool) use ImageNet pretrained weights or not
ema : Ture # (bool) exponential moving average

resume : # ckpt/fine_resnet18_pretrained.pth # resume training from giving dir, uncomment to test
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
* example: [_scripts/local.sh_](scripts/local.sh)
  * fine stage
  ```shell
  # fine stage
  export cfg_dir=cfgs/fine_resnet18_pretrained.yaml # config file dir
  export ngpu=1 bs=64 ddp=~ # set number of gpus, batchsize per gpu, distributed mode (normal, slurm or ~ (no ddp))
  export seed=42 # reset random seed for reproduction
  python fine.py
  ```
  * rough stage
  ```shell
  # rough stage
  export cfg_dir=cfgs/rough_resnet18_pretrained.yaml
  export ngpu=1 bs=64 ddp=~
  export seed=42
  python rough.py
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

### 1. put test data under _data_
* example: _data/obs/test_
```text
├─data
│  └─obs
│      └─train
|          ├─0102
|          │      person_0000.jpg
|          │      person_0001.jpg
|          │      ...
|          │      
|          ├─0103
|          │      person_0000.jpg
|          │      person_0001.jpg
|          │      ...
|          ...
│      └─test
|          ├─0102
|          │      person_xxxx.jpg
|          │      ...
|          │      
|          ├─0103
|          │      person_xxxx.jpg
|          │      ...
|          ...
```
### 2. modify _data_ and _resume_ in config files
* example: [_cfgs/fine_resnet18_pretrained.yaml_](cfgs/fine_resnet18_pretrained.yaml)
```yaml
...
data : data/obs/test # testset dir
...
resume : ckpt/fine_resnet18_pretrained.pth # resume training from giving dir, uncomment to test
...
```

### 2. modify _xxx.py_ to _xxx_test.py_ in bash script
* example: [_scripts/local.sh_](scripts/local.sh)
  * fine stage
  ```shell
  ...
  python fine_test.py
  ```
  * rough stage
  ```shell
  ...
  python rough_test.py
  ```
### 3. run script
```shell
sh scripts/local.sh
```