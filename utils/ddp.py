import os
import subprocess

import torch
import torch.distributed as dist

torch.multiprocessing.set_start_method('spawn')


# https://github.com/BIGBALLON/distribuuuu/blob/1e27266b0609e9737ed5c61902915f043b749556/tutorial/mnmc_ddp_slurm.py
def setup_slurm_distributed(cfg, backend="nccl", port=None):
    """Initialize slurm distributed training environment. (from mmcv)"""
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
    os.environ["RANK"] = str(proc_id)
    setup_distributed(cfg, backend=backend, port=port)


def setup_distributed(cfg, backend="nccl", port=None):
    # specify master port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        os.environ["MASTER_PORT"] = "29500"
    cfg.local_rank = int(os.environ["LOCAL_RANK"])
    cfg.world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(cfg.rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend)
    cfg.device = torch.device("cuda", cfg.local_rank)


"""
usage:
>>> srun --help
example:
>>> srun --partition=openai -n8 --gres=gpu:8 --ntasks-per-node=8 --job-name=slrum_test \
    python -u mnmc_ddp_slurm.py
            =======  Training  ======= 
[init] == local rank: 1, global rank: 1 ==
[init] == local rank: 7, global rank: 7 ==
[init] == local rank: 4, global rank: 4 ==
[init] == local rank: 2, global rank: 2 ==
[init] == local rank: 0, global rank: 0 ==
[init] == local rank: 5, global rank: 5 ==
[init] == local rank: 6, global rank: 6 ==
[init] == local rank: 3, global rank: 3 ==
   == step: [ 25/25] [0/5] | loss: 1.934 | acc: 29.152%
   == step: [ 25/25] [1/5] | loss: 1.546 | acc: 42.976%
   == step: [ 25/25] [2/5] | loss: 1.418 | acc: 48.064%
   == step: [ 25/25] [3/5] | loss: 1.322 | acc: 51.728%
   == step: [ 25/25] [4/5] | loss: 1.219 | acc: 55.920%
            =======  Training Finished  =======
"""
