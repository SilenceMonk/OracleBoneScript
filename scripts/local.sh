export cfg_dir=cfgs/fine_resnet18_pretrained.yaml # config file dir
export ngpu=1 bs=64 ddp=~ # number of gpus, batchsize per gpu, ddp mode (normal, slurm or ~)
export seed=42 # reset random seed for reproduction
python fine.py