export cfg_dir=cfgs/fine_resnet18_pretrained.yaml
export ngpu=4 bs=16 grad_accum_steps=1 ddp=slurm
export seed=42
srun --partition=2080ti --gres=gpu:$ngpu --ntasks-per-node=$ngpu \
python -u fine.py