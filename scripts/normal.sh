export cfg_dir=cfgs/fine_resnet18_pretrained.yaml
export ngpu=4 bs=16 ddp=normal
export seed=42
export CUDA_VISIBLE_DEVICES=0,1,2,3 # available devices
python -m torch.distributed.launch --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=localhost \
    --master_port=22222 \
    fine.py